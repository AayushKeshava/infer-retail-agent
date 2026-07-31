from __future__ import annotations

from pathlib import Path
from typing import Any

from livekit.agents import RunContext, function_tool
from livekit.agents.llm import Toolset
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.environment import get_environment


class RetailToolset(Toolset):
    """Stateful LiveKit wrappers around all tau2 Retail tools.

    Authentication state is kept per Toolset instance, so create one instance
    for each Assistant/session.
    """

    def __init__(self, db_path: str | Path) -> None:
        super().__init__(id="tau2-retail-tools")
        db = RetailDB.load(Path(db_path))
        self.env = get_environment(db=db)
        self.authenticated_user_id: str | None = None

    @property
    def policy(self) -> str:
        """Return the official tau2 Retail policy."""
        return self.env.get_policy()

    @staticmethod
    def _normalize_order_id(order_id: str) -> str:
        normalized = order_id.strip().upper().replace(" ", "")
        return normalized if normalized.startswith("#") else f"#{normalized}"

    def _serialize(self, value: Any) -> str:
        return self.env.to_json_str(value)

    def _require_authenticated_user(self) -> str:
        if self.authenticated_user_id is None:
            raise ValueError("The customer must be authenticated first.")
        return self.authenticated_user_id

    def _get_owned_order(self, order_id: str):
        user_id = self._require_authenticated_user()
        normalized = self._normalize_order_id(order_id)
        order = self.env.use_tool("get_order_details", order_id=normalized)

        if order.user_id != user_id:
            raise ValueError(
                "That order does not belong to the authenticated customer."
            )

        return normalized, order

    @staticmethod
    def _error(prefix: str, exc: ValueError) -> str:
        return f"{prefix}: {exc}"

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    @function_tool()
    async def find_user_id_by_email(
        self,
        context: RunContext,
        email: str,
    ) -> str:
        """Authenticate a retail customer using their email address.

        Args:
            email: The customer's email address.
        """
        self.authenticated_user_id = None
        try:
            user_id = self.env.use_tool(
                "find_user_id_by_email",
                email=email.strip(),
            )
            self.authenticated_user_id = user_id
            return "Authentication successful."
        except ValueError as exc:
            return self._error("Authentication failed", exc)

    @function_tool()
    async def find_user_id_by_name_zip(
        self,
        context: RunContext,
        first_name: str,
        last_name: str,
        zip: str,
    ) -> str:
        """Authenticate a customer using first name, last name, and ZIP code.

        Use this only when the customer cannot provide their email or email
        authentication fails.

        Args:
            first_name: The customer's first name.
            last_name: The customer's last name.
            zip: The customer's ZIP or postal code.
        """
        self.authenticated_user_id = None
        try:
            user_id = self.env.use_tool(
                "find_user_id_by_name_zip",
                first_name=first_name.strip(),
                last_name=last_name.strip(),
                zip=zip.strip(),
            )
            self.authenticated_user_id = user_id
            return "Authentication successful."
        except ValueError as exc:
            return self._error("Authentication failed", exc)

    # ------------------------------------------------------------------
    # Customer and order reads
    # ------------------------------------------------------------------

    @function_tool()
    async def get_user_details(self, context: RunContext) -> str:
        """Get the authenticated customer's profile, payment methods, and order IDs."""
        try:
            user_id = self._require_authenticated_user()
            result = self.env.use_tool("get_user_details", user_id=user_id)
            return self._serialize(result)
        except ValueError as exc:
            return self._error("Unable to retrieve customer details", exc)

    @function_tool()
    async def list_my_orders(self, context: RunContext) -> str:
        """List the authenticated customer's orders with status and item details.

        Use this when the customer does not know an order ID or identifies an
        order by its items or status. The database has no order timestamps, so
        do not claim which order is the most recent.
        """
        try:
            user_id = self._require_authenticated_user()
            user = self.env.use_tool("get_user_details", user_id=user_id)

            orders: list[dict[str, Any]] = []
            for order_id in user.orders:
                order = self.env.use_tool(
                    "get_order_details",
                    order_id=order_id,
                )
                orders.append(
                    {
                        "order_id": order.order_id,
                        "status": order.status,
                        "items": [
                            {
                                "name": item.name,
                                "product_id": item.product_id,
                                "item_id": item.item_id,
                                "options": item.options,
                            }
                            for item in order.items
                        ],
                    }
                )

            return self._serialize(
                {
                    "note": (
                        "Order timestamps are unavailable; do not infer which "
                        "order is the most recent."
                    ),
                    "orders": orders,
                }
            )
        except ValueError as exc:
            return self._error("Unable to list orders", exc)

    @function_tool()
    async def get_order_details(
        self,
        context: RunContext,
        order_id: str,
    ) -> str:
        """Get the status and details of an authenticated customer's order.

        Args:
            order_id: The order ID, for example W9000003.
        """
        try:
            _, order = self._get_owned_order(order_id)
            return self._serialize(order)
        except ValueError as exc:
            return self._error("Unable to retrieve order", exc)

    # ------------------------------------------------------------------
    # Product reads
    # ------------------------------------------------------------------

    @function_tool()
    async def list_all_product_types(self, context: RunContext) -> str:
        """List all product names and their product IDs."""
        try:
            return self.env.use_tool("list_all_product_types")
        except ValueError as exc:
            return self._error("Unable to list product types", exc)

    @function_tool()
    async def get_product_details(
        self,
        context: RunContext,
        product_id: str,
    ) -> str:
        """Get all variants, options, prices, and availability for a product.

        Args:
            product_id: The product ID, not an item or variant ID.
        """
        try:
            result = self.env.use_tool(
                "get_product_details",
                product_id=product_id.strip(),
            )
            return self._serialize(result)
        except ValueError as exc:
            return self._error("Unable to retrieve product", exc)

    @function_tool()
    async def get_item_details(
        self,
        context: RunContext,
        item_id: str,
    ) -> str:
        """Get options, price, and availability for one item variant.

        Args:
            item_id: The item or variant ID, not a product ID.
        """
        try:
            result = self.env.use_tool(
                "get_item_details",
                item_id=item_id.strip(),
            )
            return self._serialize(result)
        except ValueError as exc:
            return self._error("Unable to retrieve item", exc)

    # ------------------------------------------------------------------
    # Pending-order writes
    # ------------------------------------------------------------------

    @function_tool()
    async def cancel_pending_order(
        self,
        context: RunContext,
        order_id: str,
        reason: str,
    ) -> str:
        """Cancel a pending order after explicit customer confirmation.

        Args:
            order_id: The pending order ID.
            reason: Either 'no longer needed' or 'ordered by mistake'.
        """
        try:
            normalized, _ = self._get_owned_order(order_id)
            result = self.env.use_tool(
                "cancel_pending_order",
                order_id=normalized,
                reason=reason.strip().lower(),
            )
            return self._serialize(result)
        except ValueError as exc:
            return self._error("Cancellation failed", exc)

    @function_tool()
    async def modify_pending_order_address(
        self,
        context: RunContext,
        order_id: str,
        address1: str,
        address2: str,
        city: str,
        state: str,
        country: str,
        zip: str,
    ) -> str:
        """Change the shipping address of a pending order after confirmation.

        Args:
            order_id: The pending order ID.
            address1: The first address line.
            address2: The second address line, or an empty string.
            city: The city.
            state: The state or province.
            country: The country.
            zip: The ZIP or postal code.
        """
        try:
            normalized, _ = self._get_owned_order(order_id)
            result = self.env.use_tool(
                "modify_pending_order_address",
                order_id=normalized,
                address1=address1.strip(),
                address2=address2.strip(),
                city=city.strip(),
                state=state.strip(),
                country=country.strip(),
                zip=zip.strip(),
            )
            return self._serialize(result)
        except ValueError as exc:
            return self._error("Order-address modification failed", exc)

    @function_tool()
    async def modify_pending_order_items(
        self,
        context: RunContext,
        order_id: str,
        item_ids: list[str],
        new_item_ids: list[str],
        payment_method_id: str,
    ) -> str:
        """Replace variants in a pending order after explicit confirmation.

        Gather every requested item change before calling because the operation
        can be performed only once for an order.

        Args:
            order_id: The pending order ID.
            item_ids: Existing item IDs to replace.
            new_item_ids: Replacement item IDs in the same order.
            payment_method_id: Payment method for any charge or refund difference.
        """
        try:
            normalized, _ = self._get_owned_order(order_id)
            result = self.env.use_tool(
                "modify_pending_order_items",
                order_id=normalized,
                item_ids=item_ids,
                new_item_ids=new_item_ids,
                payment_method_id=payment_method_id.strip(),
            )
            return self._serialize(result)
        except ValueError as exc:
            return self._error("Order-item modification failed", exc)

    @function_tool()
    async def modify_pending_order_payment(
        self,
        context: RunContext,
        order_id: str,
        payment_method_id: str,
    ) -> str:
        """Change the payment method of a pending order after confirmation.

        Args:
            order_id: The pending order ID.
            payment_method_id: A different payment method belonging to the customer.
        """
        try:
            normalized, _ = self._get_owned_order(order_id)
            result = self.env.use_tool(
                "modify_pending_order_payment",
                order_id=normalized,
                payment_method_id=payment_method_id.strip(),
            )
            return self._serialize(result)
        except ValueError as exc:
            return self._error("Payment-method modification failed", exc)

    # ------------------------------------------------------------------
    # Delivered-order writes
    # ------------------------------------------------------------------

    @function_tool()
    async def return_delivered_order_items(
        self,
        context: RunContext,
        order_id: str,
        item_ids: list[str],
        payment_method_id: str,
    ) -> str:
        """Return items from a delivered order after explicit confirmation.

        Gather all return items before calling.

        Args:
            order_id: The delivered order ID.
            item_ids: Item IDs to return; repeat an ID for multiple quantities.
            payment_method_id: The original payment method or an eligible gift card.
        """
        try:
            normalized, _ = self._get_owned_order(order_id)
            result = self.env.use_tool(
                "return_delivered_order_items",
                order_id=normalized,
                item_ids=item_ids,
                payment_method_id=payment_method_id.strip(),
            )
            return self._serialize(result)
        except ValueError as exc:
            return self._error("Return request failed", exc)

    @function_tool()
    async def exchange_delivered_order_items(
        self,
        context: RunContext,
        order_id: str,
        item_ids: list[str],
        new_item_ids: list[str],
        payment_method_id: str,
    ) -> str:
        """Exchange delivered items for variants of the same products.

        Gather every requested exchange and obtain explicit confirmation before
        calling because the operation can be performed only once per order.

        Args:
            order_id: The delivered order ID.
            item_ids: Existing item IDs to exchange.
            new_item_ids: Replacement item IDs in the same order.
            payment_method_id: Payment method for any charge or refund difference.
        """
        try:
            normalized, _ = self._get_owned_order(order_id)
            result = self.env.use_tool(
                "exchange_delivered_order_items",
                order_id=normalized,
                item_ids=item_ids,
                new_item_ids=new_item_ids,
                payment_method_id=payment_method_id.strip(),
            )
            return self._serialize(result)
        except ValueError as exc:
            return self._error("Exchange request failed", exc)

    # ------------------------------------------------------------------
    # Customer-account write
    # ------------------------------------------------------------------

    @function_tool()
    async def modify_user_address(
        self,
        context: RunContext,
        address1: str,
        address2: str,
        city: str,
        state: str,
        country: str,
        zip: str,
    ) -> str:
        """Change the authenticated customer's default address after confirmation.

        This does not change the address of an existing order.

        Args:
            address1: The first address line.
            address2: The second address line, or an empty string.
            city: The city.
            state: The state or province.
            country: The country.
            zip: The ZIP or postal code.
        """
        try:
            user_id = self._require_authenticated_user()
            result = self.env.use_tool(
                "modify_user_address",
                user_id=user_id,
                address1=address1.strip(),
                address2=address2.strip(),
                city=city.strip(),
                state=state.strip(),
                country=country.strip(),
                zip=zip.strip(),
            )
            return self._serialize(result)
        except ValueError as exc:
            return self._error("Default-address modification failed", exc)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @function_tool()
    async def calculate(
        self,
        context: RunContext,
        expression: str,
    ) -> str:
        """Calculate arithmetic needed to complete a retail support request.

        Args:
            expression: Numbers, arithmetic operators, parentheses, and spaces.
        """
        try:
            return self.env.use_tool(
                "calculate",
                expression=expression.strip(),
            )
        except ValueError as exc:
            return self._error("Calculation failed", exc)

    @function_tool()
    async def transfer_to_human_agents(
        self,
        context: RunContext,
        summary: str,
    ) -> str:
        """Transfer to a human when requested or when policy and tools cannot solve the issue.

        Args:
            summary: A concise summary of the issue and actions already taken.
        """
        try:
            return self.env.use_tool(
                "transfer_to_human_agents",
                summary=summary.strip(),
            )
        except ValueError as exc:
            return self._error("Transfer failed", exc)