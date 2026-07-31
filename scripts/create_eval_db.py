import json
import os
from copy import deepcopy
from pathlib import Path

from tau2.domains.retail.data_model import RetailDB


source = (
    Path(os.environ["TAU2_DATA_DIR"])
    / "tau2/domains/retail/db.json"
)
output = Path("data/retail_eval_db.json")

db = json.loads(source.read_text())

# 1. Add a second order to Noah
noah = db["users"]["noah_brown_6181"]

new_order = deepcopy(db["orders"]["#W6750959"])
new_order["order_id"] = "#W9000001"
new_order["user_id"] = noah["user_id"]
new_order["address"] = deepcopy(noah["address"])

for payment in new_order["payment_history"]:
    payment["payment_method_id"] = "paypal_5727330"

db["orders"]["#W9000001"] = new_order
noah["orders"].append("#W9000001")


# 2. Add another customer with two orders
user_id = "maya_shah_eval"
payment_id = "paypal_eval_9001"

maya = {
    "user_id": user_id,
    "name": {
        "first_name": "Maya",
        "last_name": "Shah",
    },
    "address": {
        "address1": "100 Market Street",
        "address2": "Apartment 12",
        "city": "San Francisco",
        "country": "USA",
        "state": "CA",
        "zip": "94105",
    },
    "email": "maya.shah.eval@example.com",
    "payment_methods": {
        payment_id: {
            "source": "paypal",
            "id": payment_id,
        }
    },
    "orders": ["#W9000002", "#W9000003"],
}

for source_id, new_id in [
    ("#W7678072", "#W9000002"),
    ("#W6750959", "#W9000003"),
]:
    order = deepcopy(db["orders"][source_id])
    order["order_id"] = new_id
    order["user_id"] = user_id
    order["address"] = deepcopy(maya["address"])

    for payment in order["payment_history"]:
        payment["payment_method_id"] = payment_id

    db["orders"][new_id] = order

db["users"][user_id] = maya

output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(db, indent=2))

RetailDB.load(output)  # Validate schema

print(f"Created {output}")
print("Noah:", noah["orders"])
print("Maya:", maya["orders"])