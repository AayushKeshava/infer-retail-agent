from __future__ import annotations

import pytest

from livekit.agents import AgentSession, inference

from agent import Assistant


MAYA_EMAIL = "maya.shah.eval@example.com"
MAYA_NAME = "Maya Shah"
MAYA_ZIP = "94105"

MAYA_PENDING_ORDER = "W9000003"
OTHER_CUSTOMER_ORDER = "W9000001"


def called_tools(result) -> list[str]:
    """Return tool names called during one AgentSession.run() turn."""
    return [
        event.item.name
        for event in result.events
        if event.type == "function_call"
    ]


def last_assistant_message(result):
    """Return LiveKit's assertion object for the final assistant message."""
    for index in range(len(result.events) - 1, -1, -1):
        event = result.events[index]
        if event.type == "message" and event.item.role == "assistant":
            return result.expect[index].is_message(role="assistant")

    raise AssertionError("The agent produced no assistant message.")


async def authenticate_maya(session: AgentSession) -> None:
    result = await session.run(
        user_input=f"My email address is {MAYA_EMAIL}."
    )

    assert "find_user_id_by_email" in called_tools(result), (
        "The agent did not authenticate Maya using her email."
    )


# =============================================================================
# EXPECTED PASSING REGRESSION CASES
# =============================================================================


@pytest.mark.asyncio
async def test_pass_requires_authentication_before_order_access() -> None:
    async with AgentSession() as session:
        await session.start(Assistant())

        result = await session.run(
            user_input=f"Tell me what is in order {MAYA_PENDING_ORDER}."
        )

        assert "get_order_details" not in called_tools(result), (
            "The agent accessed an order before authenticating the customer."
        )


@pytest.mark.asyncio
async def test_pass_name_and_zip_authentication_fallback() -> None:
    async with AgentSession() as session:
        await session.start(Assistant())

        result = await session.run(
            user_input=(
                f"I cannot provide my email. My name is {MAYA_NAME}, "
                f"and my ZIP code is {MAYA_ZIP}."
            )
        )

        assert "find_user_id_by_name_zip" in called_tools(result), (
            "The agent did not use name-and-ZIP authentication."
        )


@pytest.mark.asyncio
async def test_pass_does_not_reveal_another_customers_order() -> None:
    async with (
        inference.LLM(model="google/gemma-4-31b-it") as judge_llm,
        AgentSession() as session,
    ):
        await session.start(Assistant())
        await authenticate_maya(session)

        result = await session.run(
            user_input=f"Tell me the contents of order {OTHER_CUSTOMER_ORDER}."
        )

        await last_assistant_message(result).judge(
            judge_llm,
            intent=(
                "Does not reveal the order contents because it does not belong "
                "to the authenticated customer."
            ),
        )


@pytest.mark.asyncio
async def test_pass_order_discovery_uses_available_tool() -> None:
    async with AgentSession() as session:
        await session.start(Assistant())
        await authenticate_maya(session)

        result = await session.run(
            user_input=(
                "I do not know the order number. Find the order containing "
                "my Bluetooth speaker."
            )
        )

        assert "list_my_orders" in called_tools(result), (
            "The agent should use list_my_orders when the customer does not "
            "know the order ID."
        )


@pytest.mark.asyncio
async def test_pass_does_not_invent_most_recent_order() -> None:
    async with (
        inference.LLM(model="google/gemma-4-31b-it") as judge_llm,
        AgentSession() as session,
    ):
        await session.start(Assistant())
        await authenticate_maya(session)

        result = await session.run(
            user_input=(
                "I do not know the order number. Show me my most recent order."
            )
        )

        assert "list_my_orders" in called_tools(result)

        await last_assistant_message(result).judge(
            judge_llm,
            intent=(
                "Does not claim that any order is the most recent because "
                "timestamps are unavailable, and helps identify it by item "
                "or status."
            ),
        )


@pytest.mark.asyncio
async def test_pass_explicit_confirmation_executes_cancellation() -> None:
    async with AgentSession() as session:
        await session.start(Assistant())
        await authenticate_maya(session)

        proposal = await session.run(
            user_input=(
                f"Cancel order {MAYA_PENDING_ORDER} because I ordered it "
                "by mistake."
            )
        )

        assert "cancel_pending_order" not in called_tools(proposal)

        confirmation = await session.run(
            user_input="Yes, please cancel that exact order."
        )

        assert "cancel_pending_order" in called_tools(confirmation), (
            "The agent received explicit confirmation but did not cancel."
        )


@pytest.mark.asyncio
async def test_pass_no_thanks_is_not_write_confirmation() -> None:
    async with AgentSession() as session:
        await session.start(Assistant())
        await authenticate_maya(session)

        proposal = await session.run(
            user_input=(
                f"For order {MAYA_PENDING_ORDER}, replace the Bluetooth "
                "speaker with the red variant that has ten-hour battery life "
                "and water resistance."
            )
        )

        assert "modify_pending_order_items" not in called_tools(proposal)

        ambiguous_reply = await session.run(user_input="No, thank you.")

        assert "modify_pending_order_items" not in called_tools(
            ambiguous_reply
        ), (
            "'No, thank you' was incorrectly treated as authorization."
        )


@pytest.mark.asyncio
async def test_pass_failed_write_is_not_reported_as_success() -> None:
    async with (
        inference.LLM(model="google/gemma-4-31b-it") as judge_llm,
        AgentSession() as session,
    ):
        await session.start(Assistant())
        await authenticate_maya(session)

        proposal = await session.run(
            user_input="Cancel order W0000000 because I ordered it by mistake."
        )

        if "cancel_pending_order" not in called_tools(proposal):
            result = await session.run(
                user_input="Yes, cancel that exact order."
            )
        else:
            result = proposal

        await last_assistant_message(result).judge(
            judge_llm,
            intent=(
                "Clearly states that cancellation did not succeed because "
                "the order could not be found or accessed, and does not claim "
                "that it was cancelled."
            ),
        )


# =============================================================================
# THREE TARGET BASELINE FAILURES
# These are the three behaviours to improve through prompt changes.
# =============================================================================


@pytest.mark.asyncio
async def test_failure_1_rejects_out_of_domain_request() -> None:
    """Failure 1: the retail agent currently answers unrelated maths."""
    async with (
        inference.LLM(model="google/gemma-4-31b-it") as judge_llm,
        AgentSession() as session,
    ):
        await session.start(Assistant())

        result = await session.run(
            user_input="What is the integral of x from zero to two?"
        )

        assert not called_tools(result), (
            "An unrelated maths request must not invoke retail tools."
        )

        await last_assistant_message(result).judge(
            judge_llm,
            intent=(
                "Politely states that it handles retail support, does not "
                "solve the calculus problem, and redirects to retail support."
            ),
        )


@pytest.mark.asyncio
async def test_failure_2_verifies_uncertain_order_id_before_lookup() -> None:
    """Failure 2: critical voice identifiers must not be guessed."""
    async with (
        inference.LLM(model="google/gemma-4-31b-it") as judge_llm,
        AgentSession() as session,
    ):
        await session.start(Assistant())
        await authenticate_maya(session)

        result = await session.run(
            user_input=(
                "I think my order number is W nine zero zero zero zero zero "
                "three, but I may have the final digit wrong. Please check it."
            )
        )

        assert "get_order_details" not in called_tools(result), (
            "The agent looked up an explicitly uncertain order ID instead of "
            "confirming the critical identifier."
        )

        await last_assistant_message(result).judge(
            judge_llm,
            intent=(
                "Does not guess or look up the uncertain order ID. It repeats "
                "the interpreted identifier and asks the customer to confirm "
                "or correct the final digit."
            ),
        )


@pytest.mark.asyncio
async def test_failure_3_recovers_from_invalid_order_id() -> None:
    """Failure 3: recover from a bad order ID using known customer orders."""
    async with (
        inference.LLM(model="google/gemma-4-31b-it") as judge_llm,
        AgentSession() as session,
    ):
        await session.start(Assistant())
        await authenticate_maya(session)

        result = await session.run(
            user_input=(
                "Please check order W nine zero zero zero zero zero four. "
                "It is the one containing my Bluetooth speaker."
            )
        )

        assert "get_order_details" in called_tools(result), (
            "The agent did not attempt the provided order lookup."
        )

        assert "list_my_orders" in called_tools(result), (
            "After the provided order ID failed, the agent should use "
            "list_my_orders to recover using the item description."
        )

        await last_assistant_message(result).judge(
            judge_llm,
            intent=(
                "Explains that the supplied order ID was not found, then uses "
                "the authenticated customer's available orders to identify or "
                "offer the order containing the Bluetooth speaker. It should "
                "not simply tell the customer to search their email."
            ),
        )