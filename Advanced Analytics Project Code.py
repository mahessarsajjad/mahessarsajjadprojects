#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 11:17:41 2023

@author: wenke
"""

import pandas as pd
import math as math
import numpy as np
import random as random
import pulp as pl


#######=================Parameter setting=================######
random.seed(1)

###### parameters in greedy adaptive
# how many customers are randomly selected for every iteration.
# from the candidate list, one customer will be selected randomly and be allocated to the best facility
GA_candidates_list_size = 12

###### parameters in first improvement
# how many customers are randomly selected for every iteration. 
# In every iteration, every customer in the list would be checked for either simple reallocation or swapping.
FI_random_list_size = 45
# the maximum iteration times.
# The iteration will stop when no improvement or no reallocation happened in the last iteration.
# However if there are alrealy more than FI_iteration_times iteration, it will stop anyway.
FI_iteration_times = 50


###### senstivity analysis 
# Whether consider the constraint 5 and 6 in OM and FI, 1 means yes, 0 means 0
Constraint56 = 1
# The percentage of capacity relaxation, e.g. 0.1 means 10% more than original capacity
Capacity_relaxation = 0
# Sepcify what initial solution is to be improved by OM and FA, "Simulation", "Sequential", "Adaptive"
Solution_for_improvement = "Adaptive"


###### whether want to run optimisation model, 1 means yes, 0 means 0
Run_OM = 1
###### whether want to run first imporvement model, 1 means yes, 0 means 0
Run_FI = 1



#######=================Data import=================######

file_name = "/Users/wenke/Desktop/AAMA-Group Assignment/customer_allocation.xlsx"

Capacity = pd.read_excel(file_name, "Capacity")
Facility = pd.read_excel(file_name, "Facility")
Customer = pd.read_excel(file_name, "Customer")
Demand = pd.read_excel(file_name, "Demand")
# in our project, we simply consider reallocation cost to be zero because it's one time and intangible
Reallocation_cost = 0


#######=================Distance calculation=================######

# create an empty arrary to save the distance calculation result
Distance = np.zeros(shape=(Customer.shape[0],Facility.shape[0]))

# calculate the euclidan distance between facility and customer
for i in range(Customer.shape[0]):
    for j in range(Facility.shape[0]):
        Distance[i,j] = math.sqrt((Customer.iloc[i,1]-Facility.iloc[j,1])**2 + (Customer.iloc[i,2]-Facility.iloc[j,2])**2)


#######=================Initial Simulation=================######        

# create an empty arrary to save the reality simulation solution
Solution_Simulation = np.zeros(shape=(Customer.shape[0],Facility.shape[0]))

# create a copy of capacity for iteration
Current_capacity =  Capacity.copy()


for i in range(Customer.shape[0]):
    
    # rank the facility by the distance to customer i 
    Rank_of_facility = pd.Series(Distance[i]).sort_values().index
    
    # generate a list of the top 2 closest facilities
    Possible_facility_list=[]
    for j in Rank_of_facility:
        if Current_capacity.iloc[j,1] > Demand.iloc[i,1]:
            if len(Possible_facility_list) <=2:
                Possible_facility_list.append(j)
    
    # randomly allocated i to one of them (the top 2 closest facilities)
    j = random.choice(Possible_facility_list)
    Solution_Simulation[i,j] = 1
    
    # update the current capacity after allocation
    Current_capacity.iloc[j,1] = Current_capacity.iloc[j,1] - Demand.iloc[i,1]


# calculate the total cost of the reality simulation solution
Total_cost_Simulation = round(sum(np.dot(Demand.iloc[:,1],Solution_Simulation*Distance)),2)                  
print("Total_cost_Simulation = ",  Total_cost_Simulation)



#######=================Greedy Sequential=================######        

# create an empty arrary to save the greddy sequential solution
Solution_GS = np.zeros(shape=(Customer.shape[0],Facility.shape[0]))

# create a copy of capacity for iteration
Current_capacity =  Capacity.copy()

for i in range(Customer.shape[0]):
    
    # rank the facility by the distance to customer i
    Rank_of_facility = pd.Series(Distance[i]).sort_values().index
    
    # allocate customer i to the closest available facility 
    for j in Rank_of_facility:
        if Current_capacity.iloc[j,1] > Demand.iloc[i,1]:
            Solution_GS[i,j] = 1
            Current_capacity.iloc[j,1] = Current_capacity.iloc[j,1] - Demand.iloc[i,1]
            break


# calculate the total cost of the greddy sequential solution
Total_cost_GS = round(sum(np.dot(Demand.iloc[:,1],Solution_GS*Distance)),2)                  
print("Total_cost_GS = ",  Total_cost_GS)



#######=================Greedy Adaptive=================######

# create an empty arrary to save the greddy adaptive solution
Solution_GA = np.zeros(shape=(Customer.shape[0],Facility.shape[0]))

# create a copy of capacity for iteration
Current_capacity =  Capacity.copy()

# create a list of customers not allocated
Customers_not_allocated = list(range(Customer.shape[0]))


while len(Customers_not_allocated) != 0:
    
    # calculate the possible cost increase of allocation to current solution for each customer
    Possible_cost_increase = np.zeros(shape=(3,len(Customers_not_allocated)))
    k = 0                                  
    for i in Customers_not_allocated:
        
        # rank the facility by the distance to customer i
        Rank_of_facility = pd.Series(Distance[i]).sort_values().index
        for j in Rank_of_facility:
            
            # choose the closest available facility j
            if Current_capacity.iloc[j,1] > Demand.iloc[i,1]:
                
                # calculate possible cost increase = distance[i,j] * demand[i]
                Possible_cost_increase[0,k] = i
                Possible_cost_increase[1,k] = j
                Possible_cost_increase[2,k] = Demand.iloc[i,1]*Distance[i,j]
                k = k+1
                break

    # rank the customer by possible cost increase from lowest to highest
    # select the top ones to generate a list of customer candidates 
    if len(Customers_not_allocated) > GA_candidates_list_size:
        Customer_candidate_list = pd.Series(Possible_cost_increase[2]).sort_values().index[1:GA_candidates_list_size+1]
    else:
        
        # if less than 10 customers are not allocated, then include all customers into candidate list
        Customer_candidate_list = pd.Series(Possible_cost_increase[2]).sort_values().index
    
    # choose a random customer i from the candidate list, allocate it to facility the best facility j
    chosen_index = random.choice(Customer_candidate_list)
    Customer_i = int(Possible_cost_increase[0,chosen_index])
    Facility_j = int(Possible_cost_increase[1,chosen_index])
    Solution_GA[int(Possible_cost_increase[0,chosen_index]),int(Possible_cost_increase[1,chosen_index])] = 1
    
    # update the list of customers not allocated
    Customers_not_allocated.remove(Customer_i)
    
    # update the current capacity of j after allocation
    Current_capacity.iloc[Facility_j,1] = Current_capacity.iloc[Facility_j,1] - Demand.iloc[Customer_i,1]


# calculate the total cost of the greddy adaptive solution
Total_cost_GA = round(sum(np.dot(Demand.iloc[:,1],Solution_GA*Distance)),2)
print("Total_cost_GA = ",  Total_cost_GA)





#######=================Optimisation Model=================######  


if Run_OM == 1:
    
    # choose what initial solution to be improved
    if Solution_for_improvement == "Simulation":
        Solution_Initial = Solution_Simulation
    
    if Solution_for_improvement == "Sequential":
        Solution_Initial = Solution_GS
    
    if Solution_for_improvement == "Adaptive":
        Solution_Initial = Solution_GA
    
    
    model = pl.LpProblem("Customer_allocation", pl.LpMinimize)
    # define variables
    Allocation = pl.LpVariable.dicts("Allocation", (Customer.index,Facility.index), lowBound=0, upBound=1, cat ="Integer")
    # objective
    model += pl.lpSum([Allocation[i][j]*Distance[i,j]*Demand.iloc[i,1] for i in Customer.index for j in Facility.index])
    # allocate once constraint
    for i in Customer.index:
        model += pl.lpSum([Allocation[i][j] for j in Facility.index]) == 1, f"Customer{i}_allocated_once"
    # capacity constraint
    for j in Facility.index:
        model += pl.lpSum([Allocation[i][j]*Demand.iloc[i,1] for i in Customer.index]) <= Capacity.iloc[j,1]*(1+Capacity_relaxation), f"Facility{j}_capacity"
    
    # if consider constraints 5 and 6, i.e. consider not only the overall cost but also the cost for each customer
    if Constraint56 == 1:   
        Reallocation = pl.LpVariable.dicts("Reallocation", Customer.index, lowBound=0, upBound=1, cat ="Integer")    
        for i in Customer.index:
            model += pl.lpSum([Solution_Initial[i][j]*Allocation[i][j] for j in Facility.index]) == 1 - Reallocation[i], f"Customer{i}_reallocation"    
        for i in Customer.index:
            model += pl.lpSum([Distance[i,j]*Demand.iloc[i,1]*(Solution_Initial[i][j]-Allocation[i][j]) for j in Facility.index]) >= Reallocation[i]*Reallocation_cost, f"Customer{i}_improvement"
    
    
    # solve model, calculate the total cost of the optimisation model solution
    model.solve(pl.PULP_CBC_CMD())
    if pl.LpStatus[model.status] == "Optimal":
        Total_cost_OM = round(pl.value(model.objective),2)
    print("Total_cost_OM = ",  Total_cost_OM)
    
    
    #model.writeLP("Customer_allocation.lp")





#######=================First Improvement=================######   


if Run_FI == 1:
         
    # choose what initial solution to be improved
    if Solution_for_improvement == "Simulation":
        Solution_Initial = Solution_Simulation
    
    if Solution_for_improvement == "Sequential":
        Solution_Initial = Solution_GS
    
    if Solution_for_improvement == "Adaptive":
        Solution_Initial = Solution_GA
            
    
    # create an empty arrary to save the first improvement solution
    Solution_FI = Solution_Initial.copy()
    
    # create a copy of capacity for iteration
    Current_capacity_relaxation =  Current_capacity.copy()
    
    # if relaxation applicable, update current capacity by adding a certain precentage of original capacity
    Current_capacity_relaxation.iloc[:,1] = Current_capacity_relaxation.iloc[:,1] + Capacity.iloc[:,1]*Capacity_relaxation
    
    
    
    
    # use c to count the number of reallocations have been made
    c = 0
    # use m to count count the iteration times
    t = 0
    # use reallocation_happened to tell whether reallocation happened in each iteration
    reallocation_happened = 1
    
    
    # keep iteration until the last iteration that did not improve the solution, i.e. no reallocation happened
    # however, if the iteration time exceeds a certain number, it will stop anyway
    while reallocation_happened != 0 and t < FI_iteration_times:
        # initialise reallocation_happened to be zero
        reallocation_happened = 0
        # count the iteration times
        t = t+1
        
        # generate a random customer list of certain size
        for i in random.sample(range(Customer.shape[0]),FI_random_list_size):
            
            #initialise k = 0 for controling the break of loop
            #if k = 1 means customer i is reallocated in this iteration
            k = 0
            
            # save the index of facility i currently allocated to
            Facility_i_allocated = np.where(Solution_FI[i]==1)[0][0]
            
            # if it is not allocated to the closest facility, then keep checking for possibility of reallocation
            if  Solution_FI[i,np.argmin(Distance[i])] != 1:
                
                for j in range(Facility.shape[0]):
                    
                    # if i is already reallocated in this iteration, break the loop and go check next customer
                    if k == 1:
                        break
                    else:
                        
                        #### simple reallocation
                        # check if customer i is closer to facility j
                        if Distance[i,j] < Distance[i,Facility_i_allocated]:
                            
                            # check if facility j has enough capacity
                            if Current_capacity_relaxation.iloc[j,1] >= Demand.iloc[i,1]:
                                
                                # if both yes, then reallocate customer i to facility j
                                # and update the current capacities of both facilities
                                Current_capacity_relaxation.iloc[j,1] = Current_capacity_relaxation.iloc[j,1] - Demand.iloc[i,1]
                                Current_capacity_relaxation.iloc[Facility_i_allocated,1] = Current_capacity_relaxation.iloc[Facility_i_allocated,1] + Demand.iloc[i,1]
                                Solution_FI[i] = np.zeros(Facility.shape[0])
                                Solution_FI[i,j] = 1
                                
                                # count one more reallocation, and update k = 1, reallocation_happened = 1
                                reallocation_happened = 1
                                c = c+1
                                k = 1
                                break
                            
                            #### if simple reallocation is not feasible, then try swapping
                            else:
                                for a in range(Customer.shape[0]):
                                    
                                    # for each customer currently allocated to facility j after previous reallocation
                                    if Solution_FI[a,j] == 1:
                                        
                                        # check if a is closer to the facility i allocated to, which corresponding to constraints 5 and 6
                                        if Distance[a,j] > Distance[a,Facility_i_allocated] or Constraint56 == 0:
                                            
                                            #check if facility j has enough capacity to do swapping
                                            if Current_capacity_relaxation.iloc[j,1] + Demand.iloc[a,1] >= Demand.iloc[i,1]:
                                                
                                                #check if facility i currently allocated to has enough capacity to do swapping
                                                if Current_capacity_relaxation.iloc[Facility_i_allocated,1] + Demand.iloc[i,1] >= Demand.iloc[a,1]:
                                                    
                                                    # check if the swapping can reduce total cost
                                                    if Distance[i,Facility_i_allocated]*Demand.iloc[i,1] + Distance[a,j]*Demand.iloc[a,1] > Distance[i,j]*Demand.iloc[i,1] + Distance[a,Facility_i_allocated]*Demand.iloc[a,1] + 2*Reallocation_cost:
                                                        
                                                        # if all yes, then swap i and a
                                                        # and update the current capacities of both facilities
                                                        Current_capacity_relaxation.iloc[j,1] = Current_capacity_relaxation.iloc[j,1] + Demand.iloc[a,1] - Demand.iloc[i,1]
                                                        Current_capacity_relaxation.iloc[Facility_i_allocated,1] = Current_capacity_relaxation.iloc[Facility_i_allocated,1] + Demand.iloc[i,1] - Demand.iloc[a,1]
                                                        Solution_FI[i] = np.zeros(Facility.shape[0])
                                                        Solution_FI[i,j] = 1
                                                        Solution_FI[a] = np.zeros(Facility.shape[0])
                                                        Solution_FI[a,Facility_i_allocated] = 1

                                                        # count two more reallocation, and update k = 1, reallocation_happened = 1
                                                        reallocation_happened = 1
                                                        c = c+2
                                                        k = 1
                                                        break
            
    
    
    # calculate the total cost of the first improvement solution
    Total_cost_FI = round(sum(np.dot(Demand.iloc[:,1],Solution_FI*Distance)),2)                  
    print("Total_cost_FI = ",  Total_cost_FI)





