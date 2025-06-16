from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ProductionOrder:
    product: str
    quantity: int
    schedule: Optional[float] = None
    released: Optional[int] = None
    duedate: Optional[float] = None
    finished: Optional[bool] = None
    priority: Optional[int] = None
    process_total: Optional[int] = None
    process_finished: Optional[int] = None
    id: int = field(init=False)

    _next_id = 1

    def __post_init__(self):
        self.id = ProductionOrder._next_id
        ProductionOrder._next_id += 1

    def to_dict(self) -> dict:
        keys = [
            "product",
            "quantity",
            "schedule",
            "released",
            "duedate",
            "finished",
            "priority",
            "process_total",
            "process_finished",
            "id",
        ]

        dict_tmp = {key: self.__dict__[key] for key in keys if key in self.__dict__}
        return dict_tmp


@dataclass
class DemandOrder:
    product: str
    quantity: int
    duedate: Optional[float] = None
    arived: Optional[float] = None
    delivered: Optional[int] = None
    delivery_mode: Literal["asReady", "onDue", "instantly"]
    id: int = field(init=False)

    _next_id = 1

    def __post_init__(self):
        self.id = DemandOrder._next_id
        DemandOrder._next_id += 1

    def to_dict(self) -> dict:
        keys = [
            "product",
            "quantity",
            "duedate",
            "arived",
            "delivered",
            "id",
        ]

        dict_tmp = {key: self.__dict__[key] for key in keys if key in self.__dict__}
        return dict_tmp
