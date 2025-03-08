import simpy

class ProductionOrder:
    _wip_counter = 0

    def __init__(self):
        self.id = ProductionOrder._wip_counter
        ProductionOrder._wip_counter += 1

        self.product: str
        self.schedule: int
        self.released: int
        self.duedate: float
        self.finished: bool
        self.quantity: int
        self.priority: int
        self.process_total: int
        self.process_finished: int
        self.processes: dict

    def to_dict(self) -> dict:
        return self.__dict__

class Info:
    def __init__(
        self,
        env: simpy.Environment,
        resources: dict,
        products: dict,
    ):
        self.env = env
        self.resources = resources
        self.products = products

        self._create_process_data()
        self._create_resources_stores()
        self._create_products_stores()

    def _create_process_data(self) -> None:
        self.processes_name_list = {}
        self.processes_value_list = {}
        self.processes = {}

        for product in self.products:
            processes = self.products[product].get("processes")
            self.processes_name_list[product] = list(processes.keys())
            self.processes_value_list[product] = list(processes.values())

    def _create_resources_stores(self) -> None:
        
        self.resource_output = {}
        self.resource_input = {}
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

class Controller:
    def __init__(self, env: simpy.Environment, info: Info):
        self.env = env
        self.info = info
    
    def release_production_order(
        self,
        product,
        quantity,
        schedule = 0,
        duedate = 0,
        priority = 0,
    ):  
        
        order = self._make_production_order(
            product = product,
            quantity = quantity,
            schedule = schedule,
            duedate = duedate,
            priority = priority
        )
        
        self.env.process(self._process_production_order(order))

        
    def _make_production_order(
        self,
        product,
        quantity,
        schedule = 0,
        released = -1,
        duedate = 0,
        finished = False,
        priority = 0,
    ):

        production_order = ProductionOrder()
        production_order.product = product
        production_order.schedule = schedule
        production_order.released = released
        production_order.duedate = duedate
        production_order.finished = finished
        production_order.quantity = quantity
        production_order.priority = priority 
        production_order.process_total = len(
            self.info.products[product]["processes"]
        )
        production_order.process_finished = 0

        return production_order


    def _process_production_order(self, order: ProductionOrder):
        product = order.product
        schedule = order.schedule

        first_process = next(iter(self.info.products[product]["processes"])) 
        first_resource = self.info.products[product]["processes"][first_process]["resource"]
        if schedule > self.env.now:
            delay = schedule - self.env.now
            yield self.env.timeout(delay)
        else:
            order.released = self.env.now
                
        # Add order to first resource input
        yield self.info.resource_input[first_resource].put(order)
    
    



