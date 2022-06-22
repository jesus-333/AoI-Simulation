import numpy as np
import matplotlib.pyplot as plt

#%%

def preprocess_deterministic(N, tot_time_simulation, generation_time):
  if(generation_time <= 0): raise Exception("generation_time must be positive (N.b. generation_time is the time between the generation of 2 subsequent update)")
  
  # Finale variable to return that contain the list of waiting time and the relative label for each time
  arrival_time_list = []
  arrival_time_label = []

  t = 0
  label = 0

  while(True):
    arrival_time_list.append(t)
    arrival_time_label.append(label)

    t += generation_time
    label += 1

    if(t > tot_time_simulation * 1.1): break
    if(label >= N): label = 0

  return np.asarray(arrival_time_list), np.asarray(arrival_time_label)


def preprocess_exp(N, tot_time_simulation, scale):
  # Variable declaration

  # Finale variable to return that contain the list of waiting time and the relative label for each time
  arrival_time_list = np.zeros(0)
  arrival_time_label = np.zeros(0)

  # Vector that contain the label for new generated waiting time vector
  tmp_label = np.asarray([int(i) for i in range(N)])

  # Last N waiting time generated (1 for each sensor/node/etc)
  last_waiting_time = np.zeros(N)

  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  # Waiting time creation

  # Random generation process
  while(True):
    # Generate waiting time according to distribution
    tmp_waiting_time = np.random.exponential(scale = scale, size = N)

    # Shift new waiting time according to the previous generated waiting time for each node
    tmp_waiting_time += last_waiting_time

    # Concatenate waiting time and label
    arrival_time_list = np.concatenate((arrival_time_list, tmp_waiting_time))
    arrival_time_label = np.concatenate((arrival_time_label, tmp_label))

    # Save the last generated (and shifted) waiting time to use them in the new cycle
    last_waiting_time = tmp_waiting_time

    # Check if I have generated enough waiting time
    if(np.max(last_waiting_time) >= tot_time_simulation): break

    # print(np.max(last_waiting_time))

  # Sort indices based on waiting time
  sorted_idx = np.argsort(arrival_time_list)

  # Sort waiting time and labels
  arrival_time_list = arrival_time_list[sorted_idx]
  arrival_time_label = arrival_time_label[sorted_idx]

  return arrival_time_list, arrival_time_label.astype(int)


def preprocess_exp_V2(N, tot_time_simulation, scale):
    arrival_time_list = np.zeros(0)
    arrival_time_label = np.zeros(0)
    
    for i in range(N):
        tmp_arrival_time_list = generate_arrival_time_single_sensor(tot_time_simulation, scale * N)
        tmp_label = np.ones(len(tmp_arrival_time_list)) * i
        
        # Concatenate waiting time and label
        arrival_time_list = np.concatenate((arrival_time_list, tmp_arrival_time_list))
        arrival_time_label = np.concatenate((arrival_time_label, tmp_label))
     
    # Sort indices based on waiting time
    sorted_idx = np.argsort(arrival_time_list)

    # Sort waiting time and labels
    arrival_time_list = arrival_time_list[sorted_idx]
    arrival_time_label = arrival_time_label[sorted_idx]

    return arrival_time_list, arrival_time_label.astype(int)
    

def generate_arrival_time_single_sensor(tot_time_simulation, scale):
    arrival_time_list = [0]
    idx = 1

    while(True):
        tmp_arrival_time = np.random.exponential(scale = scale)
        # print("\t", arrival_time_list[idx - 1], tmp_arrival_time)
        
        tmp_arrival_time += arrival_time_list[idx - 1]
        arrival_time_list.append(tmp_arrival_time)
        idx += 1
        
        # print(tmp_arrival_time)
        
        if(tmp_arrival_time >= tot_time_simulation): break
    
    return np.asarray(arrival_time_list[1:])


def compute_final_aoi_per_sensor(aoi_list_per_sensor, correction_list_per_sensor = []):
  final_aoi_per_sensor = []

  for i in range(len(aoi_list_per_sensor)): 
    aoi_list = aoi_list_per_sensor[i]

    if(len(correction_list_per_sensor) != 0): correction_list = correction_list_per_sensor[i]
    else: correction_list = []

    final_aoi_per_sensor.append(compute_aoi_given_delta(aoi_list, correction_list))

  return final_aoi_per_sensor

def compute_aoi_given_delta(delta_list, correction_list = []):
    if(len(correction_list) > 0):
      numerator = 0
      for i in range(len(delta_list)): numerator += ((delta_list[i] ** 2) / 2) + correction_list[i] * delta_list[i]
    else:
      numerator = np.sum(np.power(delta_list, 2)) / 2
    
    denominator = np.mean(delta_list) * len(delta_list)
    
    aoi = numerator/denominator

    return aoi

#%%

def simulation_V2(N, tot_time_simulation, service_rate, arrival_time_list, arrival_time_label, simulation_step = -1, laura_correction = False, alpha = -1):
  """
  Function 
  """

  service_scale = 1 / service_rate

  # Variable to save packet related data
  queue = [arrival_time_label[0]] # Already insert the first element in the queue
  time_in_queue = [0]
  process_time = [np.random.exponential(scale = service_scale)]
  aoi_list_per_sensor = [[] for i in range(N)]
  current_aoi_per_sensor = np.zeros(N)

  tmp_actual_time = [0]
  tmp_evolution_aoi = [current_aoi_per_sensor]

  # Simulation step (equals to the interval needed to generate a new packet)
  if(simulation_step <= 0): simulation_step = np.min(np.abs(np.diff(arrival_time_label)))

  actual_time = 0

  # Index for the arrival_time_list and arrival_time_label
  idx_packet_generated = 1
  tot_remove = 0

  if(laura_correction): 
    laura_correction_factor_per_sensor = [[] for i in range(N)]
    laura_idx = 0
    
  if(alpha > 0 and alpha <= 1): use_correlation = True
  else: use_correlation = False

  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  # Simulation
  while(tot_time_simulation > actual_time):
    # Reduce the time of the first user in the queue (if there are user)
    if(len(queue) > 0): process_time[0] -= simulation_step

    # Iterate through the packet in the queue to evaluate how time they spent in the queue
    idx_to_remove = []
    for i in range(len(queue)):
      if(process_time[i] > 0): # If the waiting time is bigger than 0 they have to wait.
        time_in_queue[i] += simulation_step
      else: # If the waiting time is equal or less than 0 it means that the packet was processed
        # print(i, queue[i])
        
        # Calculate the difference between the time step and the actual non positive waiting time
        # (This is needed due the fact that the simulation is discrete with a predefined step)
        processing_time_adjustement = process_time[i] + simulation_step
        
        # Calculate the final amount of time spent in the system
        current_aoi_per_sensor[queue[i]] += processing_time_adjustement
        # N.b. In the queue I save the label of the current packet

        # Save the final AoI for the current packet
        aoi_list_per_sensor[queue[i]].append(current_aoi_per_sensor[queue[i]])

        # Correct time passed in the queue
        time_in_queue[i] -= processing_time_adjustement

        # Reset AoI for the current sensor
        current_aoi_per_sensor[queue[i]] = time_in_queue[i]
        if(use_correlation): # In the case with correlation with probability alpha reset the AoI of all the sensor
            if(np.random.rand(1)[0] < alpha): 
                current_aoi_per_sensor = np.ones(N) * time_in_queue[i]
                

        # Add the index to the list of element to remove
        # This is needed in case the waiting time of the next element is less or equal than the time_adjustment. 
        # In this case in the time of a single generation cycle more than 1 packet is processed 
        idx_to_remove.append(i)

        # Correct the waiting time for the next packet in the queue (if there are packet in the queue)
        if(len(queue) > i + 1): process_time[i + 1] -= processing_time_adjustement


        if(laura_correction): laura_correction_factor_per_sensor[queue[i]].append(time_in_queue[i])
        


    # Remove all packet processed in this iteration
    # print("\t", len(queue), process_time)
    for idx in sorted(idx_to_remove, reverse = True):
        del queue[idx]
        del process_time[idx]
        del time_in_queue[idx]
    
    actual_time += simulation_step

    current_aoi_per_sensor += simulation_step

    tmp_actual_time.append(actual_time)
    tmp_evolution_aoi.append(list(current_aoi_per_sensor))

    # Generate new packet(s)
    if(idx_packet_generated < len(arrival_time_label)):
      while(actual_time >= arrival_time_list[idx_packet_generated]):
        queue.append(arrival_time_label[idx_packet_generated].astype(int))
        time_in_queue.append(0)
        process_time.append(np.random.exponential(scale = service_scale))

        # If the queue is of length 1 it means that before this packet was empty
        # So I have to adjust the time that remain to spent in the queue
        if(len(queue) == 1):
          time_adjustement = actual_time - arrival_time_list[idx_packet_generated]
          process_time[0] -= time_adjustement

        idx_packet_generated += 1

      # print(actual_time, len(queue), idx_packet_generated, len(arrival_time_list))
  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  if(laura_correction): 
    return aoi_list_per_sensor, laura_correction_factor_per_sensor, tmp_actual_time, tmp_evolution_aoi
  else:
    return aoi_list_per_sensor, tmp_actual_time, tmp_evolution_aoi


#%%


# Number of sensors
N = 15


# Average rate of service for the server in second
service_rate = 1

# Total seconds to simulate
tot_time_simulation = 1000

laura_correction = True

alpha = 0.3

generation_rate = 1/(0.08 * N)
# arrival_time_list, arrival_time_label = preprocess_exp(N, tot_time_simulation, scale = 1 /generation_rate)
arrival_time_list, arrival_time_label = preprocess_exp_V2(N, tot_time_simulation, scale = 1 /generation_rate)
print(np.mean(np.diff(arrival_time_list)))

# generation_rate = 1/(0.04 * N)
# arrival_time_list, arrival_time_label = preprocess_deterministic(N, tot_time_simulation, generation_rate)

aoi_list_per_sensor, laura_correction_factor_per_sensor, tmp_actual_time, tmp_evolution_aoi = simulation_V2(N, tot_time_simulation, service_rate, arrival_time_list, arrival_time_label, simulation_step = 1/service_rate, laura_correction = laura_correction)
print("Average AoI = ", np.mean(compute_final_aoi_per_sensor(aoi_list_per_sensor)), "(No correction)")
print("Average AoI = ", np.mean(compute_final_aoi_per_sensor(aoi_list_per_sensor, laura_correction_factor_per_sensor)), "(Correction)")

# tmp_evolution_aoi = np.asarray(tmp_evolution_aoi)
# # plt.plot(aoi_list_per_sensor[2])
# plt.plot(tmp_actual_time, tmp_evolution_aoi[:, 0])
# # print(aoi_list_per_sensor)

#%%
# while(True):
#     generation_rate = 1/(0.06 * N)
#     # arrival_time_list, arrival_time_label = preprocess_exp(N, tot_time_s    imulation, scale = 1 /generation_rate)
#     arrival_time_list, arrival_time_label = preprocess_exp_V2(N, tot_time_simulation, scale = 1 /generation_rate)
#     if(np.mean(np.diff(arrival_time_list)) >= 1): break
    
a1 = []
a2 = []

for i in range(250):
    aoi_list_per_sensor, laura_correction_factor_per_sensor, tmp_actual_time, tmp_evolution_aoi = simulation_V2(N, tot_time_simulation, service_rate, arrival_time_list, arrival_time_label, simulation_step = 1/service_rate, laura_correction = laura_correction)
    a1.append(np.mean(compute_final_aoi_per_sensor(aoi_list_per_sensor)))
    a2.append(np.mean(compute_final_aoi_per_sensor(aoi_list_per_sensor, laura_correction_factor_per_sensor)))

print("Average AoI = ", np.mean(a1), "(No correction)")
print("Average AoI = ", np.mean(a2), "(Correction)")
