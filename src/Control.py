import simpy


class Stores:
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
   
    
class ProductionOrder:
    _wip_counter = 0

    def __init__(
        self,
        env: simpy.Environment,
        info: Info,
        product: str,
        quantity: int = 1,
        schedule: float = 0,
        released: int = -1,
        duedate: float = 0,
        finished: bool = False,
        priority: int = 0,
    ):
        self.id = ProductionOrder._wip_counter
        ProductionOrder._wip_counter += 1

        self.env = env
        self.info = info
        self.product = product
        self.schedule = schedule
        self.released = released
        self.duedate = duedate
        self.finished = finished
        self.quantity = quantity
        self.priority = priority 
        self.process_total = 0
        self.process_finished = 0

    def to_dict(self) -> dict:


        return self.__dict__
    

    def release(self) -> None:
        self.env.process(self._process())

    def _process(self):
        
        self.process_total = len(
            self.info.products[self.product]["processes"]
        )

        first_process = next(iter(self.info.products[self.product]["processes"])) 
        first_resource = self.info.products[self.product]["processes"][first_process]["resource"]
        if self.schedule > self.env.now:
            delay = self.schedule - self.env.now
            yield self.env.timeout(delay)
        else:
            self.released = self.env.now
                
        # Add order to first resource input
        print(self.to_dict())
        yield self.info.resource_input[first_resource].put(self.to_dict())
    


