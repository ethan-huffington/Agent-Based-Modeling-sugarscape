import mesa
import math
import numpy as np
from resource_agents import Sugar, Spice
from trader_agents import Trader


#=============================================================================
# HELPER FUNCTIONS
#=============================================================================
def get_distance(pos_1, pos_2):
    ''' Euclidean Distance between two points '''
    x1, y1 = pos_1
    x2, y2 = pos_2
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx**2 + dy**2)
    
def flatten(list_of_lists):
    ''' Collapses a list of lists into single list'''
    return [item for sublist in list_of_lists for item in sublist]

def geometric_mean(list_of_prices):
    '''Help func for logging:
    Returns geometric mean from a list of prices
    '''
    return np.exp(np.log(list_of_prices).mean())

def get_trade(agent):
    ''' Help funct for data collector '''
    if type(agent) == Trader:
        return agent.trade_partners
    else: return None

#=============================================================================
# - - - - - - - - - - - - - - - MODEL - - - - - - - - - - - - - - - - - - - - 
#=============================================================================

class SugarscapeG1mt(mesa.Model):
  '''
  A model class to manage the sugarscape with traders model
  '''
  def __init__(self, width=50, height=50, initial_population=200,
               endowment_min=25, endowment_max=50, metabolism_min=1,
               metabolism_max=5, vision_min=1, vision_max=5,
               collect_agent_data = True):

    # DIMESNION SETUP
    self.width = width
    self.height = height

    # POPULATION ATTRIBUTES SETUP
    self.initial_population = initial_population
    self.endowment_min = endowment_min
    self.endowment_max = endowment_max
    self.metabolism_min = metabolism_min
    self.metabolism_max = metabolism_max
    self.vision_min = vision_min
    self.vision_max = vision_max
    self.running = True # For batch-runs

    # MAP SETUP
    self.grid = mesa.space.MultiGrid(self.width, self.height, torus=False)
    self.sugar_distribution = np.genfromtxt("sugar-map.txt") # Read landscape .txt file
    self.spice_distribution = np.flip(self.sugar_distribution, 1)

    # SCHEDULE (TIME) SETUP
    self.schedule = mesa.time.RandomActivationByType(self)
    
    #DATA COLLECTION - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    # Model Reporters
    model_reporters = {
            "Num Traders": lambda m: m.schedule.get_type_count(Trader),
            "Trade Volume": lambda m: sum(len(a.trade_partners)
            for a in m.schedule.agents_by_type[Trader].values()),
            "Geo Avg Price": lambda m: geometric_mean(flatten(
             [a.prices for a in m.schedule.agents_by_type[Trader].values()])),
            "Total Wealth": lambda m: sum(flatten(
             [a.prices for a in m.schedule.agents_by_type[Trader].values()]))}
    
    # (Conditional) Agent Reporting
    agent_reporters = {"Trade Network": lambda a: get_trade(a)} if collect_agent_data else {}
    
    self.datacollector = mesa.DataCollector(model_reporters = model_reporters,
                                            agent_reporters = agent_reporters)
    
    #===========================================================================
    # INSTANTIATE AGENTS
    #===========================================================================
    
    agent_id = 0
    
    # SUGAR
    for cell_content, (x, y) in self.grid.coord_iter():
      max_sugar = self.sugar_distribution[x,y]
      if max_sugar > 0:
        # instantiate Agent
        sugar = Sugar(agent_id, self, (x,y), max_sugar)
        self.grid.place_agent(sugar, (x,y)) # place the agent on respective coordinate
        self.schedule.add(sugar) # place agent 'in time'
        agent_id += 1

    # SPICE
    for cell_content, (x, y) in self.grid.coord_iter():
      max_spice = self.spice_distribution[x,y]
      if max_spice > 0:
        # instantiate Agent
        spice = Spice(agent_id, self, (x,y), max_spice)
        self.grid.place_agent(spice, (x,y))
        self.schedule.add(spice)
        agent_id += 1

    # TRADERS
    for i in range (self.initial_population):
      x = self.random.randrange(self.width)
      y = self.random.randrange(self.height)
      sugar = int(self.random.uniform(self.endowment_min, self.endowment_max+1))
      spice = int(self.random.uniform(self.endowment_min, self.endowment_max+1))
      metabolism_sugar = int(self.random.uniform(self.metabolism_min, self.metabolism_max+1))
      metabolism_spice = int(self.random.uniform(self.metabolism_min, self.metabolism_max+1))
      vision = int(self.random.uniform(self.vision_min, self.vision_max+1))
      # instantiate Agent
      trader = Trader(agent_id,
                      self,
                      (x,y),
                      moore = False,
                      sugar = sugar,
                      spice = spice,
                      metabolism_sugar = metabolism_sugar,
                      metabolism_spice = metabolism_spice,
                      vision = vision)
      self.grid.place_agent(trader, (x,y))
      self.schedule.add(trader)
      agent_id += 1
  #=============================================================================
  # HELPER FUNCTIONS
  #=============================================================================
  def randomize_traders(self):
    ''' Helper func for step()'''
    trader_shuffle = list(self.schedule.agents_by_type[Trader].values()) # get list of trader ID's
    self.random.shuffle(trader_shuffle) # Shuffle that list (during every time step)
    return trader_shuffle

  #=============================================================================
  # TIME STEP DEFINITION
  #=============================================================================

  def step(self):

    # SUGAR & SPICE
    for sugar in self.schedule.agents_by_type[Sugar].values(): #Step all sugars
      sugar.step()
    for spice in self.schedule.agents_by_type[Spice].values(): #Step all spices
      spice.step()

    # TRADERS
    trader_shuffle = self.randomize_traders()

    for agent in trader_shuffle:
      agent.prices = [] # Clear previous round data
      agent.trade_partners = [] # Clear previous round data
      agent.move()
      agent.eat()
      agent.maybe_die()

    trader_shuffle = self.randomize_traders() # Refresh and reshuffle remaining traders

    for agent in trader_shuffle:
      agent.trade_with_neighbors()

    # SCHEDULE / DATA LOGGING
    self.schedule.steps += 1 # Need to sync 'time' with non-standard implimentation of step above
    self.datacollector.collect(self)

  #=============================================================================
  # RUN MODEL
  #=============================================================================
  def run_model(self, step_count=1000):

    for i in range(step_count):
      self.step()