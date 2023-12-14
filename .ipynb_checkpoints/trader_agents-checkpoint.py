import mesa
import math
import numpy as np
from resource_agents import Sugar, Spice


def get_distance(pos_1, pos_2):
    ''' Euclidean Distance between two points '''
    x1, y1 = pos_1
    x2, y2 = pos_2
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx**2 + dy**2)


class Trader(mesa.Agent):
  '''
  - Has a metabolism for sugar and spice
  '''
  def __init__(self, unique_id, model, pos, moore = False, sugar = 0,
               spice = 0, metabolism_sugar = 0, metabolism_spice = 0,
               vision = 0):

    super().__init__(unique_id, model)
    self.pos = pos
    self.moore = moore
    self.sugar = sugar
    self.spice = spice
    self.metabolism_sugar = metabolism_sugar
    self.metabolism_spice = metabolism_spice
    self.vision = vision
    self.prices = [] # For trade data logging
    self.trade_partners = [] # For trade data logging
    self.wealth = self.sugar + self.spice # For trade data logging

  #=============================================================================
  # HELPER FUNCTIONS
  #=============================================================================

  # MOVE & CONSUME - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  def get_distance(pos_1, pos_2):
    ''' Euclidean Distance between two points '''
    x1, y1 = pos_1
    x2, y2 = pos_2
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx**2 + dy**2)
  
  def is_occupied_by_other(self,pos):
    ''' For part 1 of move() '''
    # (For any given cell) See if you are standing there
    if pos == self.pos:
      return False
    # (For a given cell) See if anyone else is standing there
    this_cell = self.model.grid.get_cell_list_contents(pos)
    for a in this_cell:
      if isinstance(a,Trader):
        return True
    return False

  def get_sugar(self, pos):
    ''' Help func: get_sugar_amount(), eat(), maybe_die'''

    this_cell = self.model.grid.get_cell_list_contents(pos)
    for agent in this_cell:
      if type(agent) is Sugar:
        return agent
    return None

  def get_sugar_amount(self,pos):
    '''Helper function for calculate_welfare()'''

    sugar_patch = self.get_sugar(pos)
    if sugar_patch:
      return sugar_patch.amount
    return 0

  def get_spice(self, pos):
    ''' Help func: get_spice_amount(), eat(), maybe_die'''

    # Return Spice agents in cell (if any)
    this_cell = self.model.grid.get_cell_list_contents(pos)
    for agent in this_cell:
      if type(agent) is Spice:
        return agent
    return None

  def get_spice_amount(self,pos):
    '''Helper function for calculate_welfare()'''

    # Returned spice amounts of agents in cell
    spice_patch = self.get_spice(pos)
    if spice_patch:
      return spice_patch.amount
    return 0

  def calculate_welfare(self, sugar, spice):
    ''' For Part 2 of move() '''

    # Calculate agent resources
    m_total = self.metabolism_sugar + self.metabolism_spice
    # Agent Utility Algorithm (Cobb-Douglas Function)
    return sugar**(self.metabolism_sugar/m_total) * spice**(self.metabolism_spice/m_total)

  def is_starved(self):
    ''' Help func for maybe_die()'''

    return (self.sugar <= 0) or (self.spice <= 0)

  # TRADE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  def get_trader(self, pos):
    ''' Help func for trade_with_neighbors'''

    this_cell = self.model.grid.get_cell_list_contents(pos)

    for agent in this_cell:
      if isinstance(agent, Trader):
        return agent

  def calculate_MRS(self, sugar, spice):
    '''
     Help func for trade() -> trade_with_neighbors
    - Functions as a 'spice need / sugar need' ratio
    '''
    return(spice/self.metabolism_spice) / (sugar/self.metabolism_sugar)

  def calculate_sell_spice_amount(self, price):
    ''' Help func for maybe_sell_spice()->trade()->trade_with_neighbors '''
    if price >= 1:
      sugar = 1
      spice = int(price)
    else:
      sugar = int(1/price)
      spice = 1
    return sugar, spice

  def sell_spice(self, other, sugar, spice):
    ''' Help func for maybe_sell_spice()->trade()->trade_with_neighbors '''
    # Execute trade exchange
    self.sugar += sugar
    other.sugar -= sugar
    self.spice -= spice
    other.spice += spice
    return

  def maybe_sell_spice(self, other, price, welfare_self, welfare_other):
    ''' Help func for trade() -> trade_with_neighbors '''

    sugar_exchanged, spice_exchanged = self.calculate_sell_spice_amount(price)

    # Computing resource amounts if the trade DID occur
    self_sugar = self.sugar + sugar_exchanged
    other_sugar = other.sugar - sugar_exchanged
    self_spice = self.spice - spice_exchanged
    other_spice = other.spice + spice_exchanged

    # Ensure both parties have resources
    if any(x <= 0 for x in [self_sugar,
                            other_sugar,
                            self_spice,
                            other_spice]):
      return False
    
    # Trade Criterion #1: Mutual Benefit
    mutual_benefit = (
     (welfare_self < self.calculate_welfare(self_sugar, self_spice)) and
     (welfare_other < other.calculate_welfare(other_sugar, other_spice)))

    # Trade Criterion #2: MRSs Not Crossing
    mrs_not_crossing = (
        self.calculate_MRS(self_sugar, self_spice) > 
        other.calculate_MRS(other_sugar, other_spice)
    )

    if not (mutual_benefit and mrs_not_crossing):
      return False

    # Go through with trade
    self.sell_spice(other, sugar_exchanged, spice_exchanged)

    return True

    # print(f" mutual_benefit: {mutual_benefit},mrs_not_crossing: {mrs_not_crossing} ")

  def trade(self, other):
    ''' Help func for trade_with_neighbors'''

    # Sanity checking the agents have something to trade
    assert self.sugar > 0
    assert self.spice > 0
    assert other.sugar > 0
    assert other.spice > 0

    # Calculate Marginal Rates of Substitution
    mrs_self = self.calculate_MRS(self.sugar, self.spice)
    mrs_other = other.calculate_MRS(self.sugar, self.spice)

    welfare_self = self.calculate_welfare(self.sugar, self.spice)
    welfare_other = other.calculate_welfare(other.sugar, other.spice)

    if math.isclose(mrs_self, mrs_other):
      return

    # Calculate 'price'
    price = math.sqrt(mrs_self*mrs_other)
    # print(f"price: {p}, MRS self: {mrs_self}, MRS other: {mrs_other}")

    if mrs_self > mrs_other:
      # Self is a sugar buyer, spice seller
      sold = self.maybe_sell_spice(other, price, welfare_self, welfare_other)
      if not sold:
        # (Role reversal) Self is sugar seller, spice buyer
        return
    else:
      sold = other.maybe_sell_spice(self, price, welfare_other, welfare_self)
      if not sold:
        return

    # Capture Data (append attribute values to Traders)
    self.prices.append(price)
    self.trade_partners.append(other.unique_id)

    # Continue trading
    self.trade(other)

  #=============================================================================
  # MAIN FUNCTIONS
  #=============================================================================

  def move(self):
    '''
    An agent move is broken down into 4 steps:

    1 - Identify possible moves
    2 - Determine optimal move (on welfare criteria)
    3 - Find closest best move option
    4 - move
    '''

    # 1 - Itentify Possible Moves
    neighbors = [i
                 for i in self.model.grid.get_neighborhood(
                     self.pos, self.moore, True, self.vision
                     ) if not self.is_occupied_by_other(i)]

    # 2 - Determine Move Welfares
    welfares = [
        self.calculate_welfare(
            self.sugar + self.get_sugar_amount(pos),
            self.spice + self.get_spice_amount(pos))
        for pos in neighbors
    ]
    # 3 - Find Closest X Highest-Welfare Options
    max_welfare = max(welfares)
    # Get index for max-welfare cells
    candidate_indices = [i for i in range(len(welfares))
                         if math.isclose(welfares[i], max_welfare)]

    # Filter neighbors (pos list) by index
    candidates = [neighbors[i] for i in candidate_indices]
    min_dist = min(get_distance(self.pos, pos) for pos in candidates)

    # Filter candidates by index
    final_candidates = [pos for pos in candidates
                        if math.isclose(get_distance(self.pos, pos), min_dist,
                                        rel_tol=1e-02)]

    # 4 - Move Agent (to first of suitable choices)
    self.model.grid.move_agent(self, final_candidates[0])

  def eat(self):

    sugar_patch = self.get_sugar(self.pos)
    if sugar_patch:
      self.sugar += sugar_patch.amount # Add sugar to trader resources
      sugar_patch.amount = 0 # Update sugar patch amount
    self.sugar -= self.metabolism_sugar

    spice_patch = self.get_spice(self.pos)
    if spice_patch:
      self.spice += spice_patch.amount # Add to trader resources
      spice_patch.amount = 0 # Update
    self.spice -= self.metabolism_spice

  def maybe_die(self):

    if self.is_starved():
      self.model.grid.remove_agent(self) # Remove from grid
      self.model.schedule.remove(self) # Remove from schedule

  def trade_with_neighbors(self):
    '''
    Finds trading partners and iterates trade()

    1 - Locate Neighbors (per vision ability)
    2 - Trade (2-step)
    3 - Collect Data
    '''
    # Get Trader Agents of all neighboring cells
    neighbor_agents = [self.get_trader(pos) # Return Trader agent
                      for pos in           # For every returned from get_neighboorhood()
                      self.model.grid.get_neighborhood(self.pos, self.moore, False, self.vision)
                      if self.is_occupied_by_other(pos)] # If the agent isn't you

    # If no one around
    if len(neighbor_agents) == 0:
      return

    for a in neighbor_agents:
      if a:
        self.trade(a)
    return

