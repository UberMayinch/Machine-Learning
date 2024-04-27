each agent is modelled as a point on a 2x2 grid. 
the agent is assigned an inherent evolutionary fitness value which is randomly chosen in some interval.

continuous grid of some size.
values are modelled as thought viruses that can spread from agent to agent.

global parameters:
no. of thought viruses
connection parameter that grows exponentially i.e it remains less for a long time but then starts shooting up and after one point the whole board basically gets connected. This models media.



initially we start with k clusters of agents.

At each time step the following happens:



The agent has the following properties:
1. Coordinates on the 2x2 grid i.e. its location in the world.
2. Evolutionary fitness - It's "skill" at navigating through the world
3. Values - an array holding information about how much an agent conforms to a certain value.  The array can be extended as new values are discovered.
4. Age - number of iterations the agent has gone through. As this value gets over a specific threshold, the chances of the agent dying keep increasing. 

The board has the following properties:
1. Connection parameter - this is a parameter that grows exponentially. Agents whose coordinate distance is less than this may be connected. 

We initalize k clusters
the chance of a value spreading to another agent is given by the evolutionary fitness of the agent
