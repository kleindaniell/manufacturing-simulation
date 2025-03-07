import simpy


class Info:
    def __init__(
        self,
        env,
        resources: dict,
        products: dict,
    ):
        self.env = env
        self.resources = resources
        self.products = products
        self._create_process_data()

    def _create_process_data(self) -> None:
        self.processes_name_list = {}
        self.processes_value_list = {}
        self.processes = {}

        for product in self.products:
            processes = self.products[product].get("processes")
            self.processes_name_list[product] = list(processes.keys())
            self.processes_value_list[product] = list(processes.values())

    def _create_resources_stores(self) -> None:
        # Queues
        self.resource_output = {}
        self.resource_input = {}
        # KPIs
        self.resource_utilization = {}
        self.resource_breakdowns = {}
        for resource in self.resources:
            self.resource_output[resource] = simpy.FilterStore(self.env)
            self.resource_input[resource] = simpy.FilterStore(self.env)
            self.resource_utilization[resource] = 0
            self.resource_breakdowns[resource] = []

    def _create_products_stores(self) -> None:
        # Outbound
        self.finished_orders = {}
        self.finished_goods = {}
        # Inbound
        self.demand_orders = {}
        # KPIs
        self.delivered_ontime = {}
        self.delivered_late = {}
        self.lost_sales = {}
        self.wip = {}
        self.total_wip = simpy.Container
        for product in self.products:
            self.finished_orders[product] = simpy.FilterStore(self.env)
            self.finished_goods[product] = simpy.Container(self.env)
            self.demand_orders[product] = simpy.FilterStore(self.env)
            self.delivered_ontime[product] = simpy.Container(self.env)
            self.delivered_late[product] = simpy.Container(self.env)
            self.lost_sales[product] = simpy.Container(self.env)
            self.wip[product] = simpy.Container(self.env)


class ProductionOrder:
    _wip_counter = 0

    def __init__(self, product, quantity, products_config):
        self.id = ProductionOrder._wip_counter
        ProductionOrder._wip_counter += 1

        self.product = product
        self.schedule = 0
        self.released = -1
        self.duedate = 0
        self.finished = False
        self.quantity = quantity
        self.priority = 0
        self.process_total = len(products_config[product]["processes"])
        self.process_finished = 0
        self.processes = {}
        for process in products_config[product]["processes"]:
            self.processes[process] = -1

    def to_dict(self) -> dict:
        return self.__dict__
