import mesa

# SUGAR
class Sugar(mesa.Agent):
  '''
  - Contains an amount of sugar
  - Grows one amount of sugar each turn
  '''
  def __init__(self, unique_id, model, pos, max_sugar):
    super().__init__(unique_id, model)
    self.pos = pos
    self.amount = max_sugar # Amount at a given time step
    self.max_sugar = max_sugar # Total amount allowed at a location

  def step(self):
    '''
    Sugar growth function, adds 1 unit sugar each turn
    '''
    self.amount = min([self.max_sugar, self.amount+1]) # This either adds 1 sugar or enforces max_sugar

# SPICE
class Spice(mesa.Agent):
  '''
  - Contains an amount of spice
  - Grows one amount of spice each turn
  '''
  def __init__(self, unique_id, model, pos, max_spice):
    super().__init__(unique_id, model)
    self.pos = pos
    self.amount = max_spice # Amount at a given time step
    self.max_spice = max_spice # Total amount allowed at a location

  def step(self):
    '''
    Spice growth function, adds 1 unit spice each turn
    '''
    self.amount = min([self.max_spice, self.amount+1]) # This either adds 1 sugar or enforces max_spice