import pandas as pd

valid_sample_data = pd.DataFrame({
    "gender": ["Male", "Female", "Male", "Female"],
    "customer_type": ["Loyal Customer", "Disloyal Customer", "Loyal Customer", "Disloyal Customer"],
    "age": [25, 45, 35, 50],
    "type_of_travel": ["Business travel", "Personal Travel", "Personal Travel", "Business travel"],
    "class": ["Business", "Eco", "Eco Plus", "Business"],
    "flight_distance": [1500, 2000, 500, 2500],
    "inflight_wifi_service": [4, 3, 5, 2],
    "time_convenient": [4, 2, 5, 3],
    "ease_of_online_booking": [5, 3, 4, 2],
    "gate_location": [3, 4, 2, 5],
    "food_and_drink": [4, 5, 3, 2],
    "online_boarding": [4, 2, 5, 3],
    "seat_comfort": [5, 3, 4, 2],
    "inflight_entertainment": [4, 3, 5, 2],
    "on_board_service": [4, 2, 5, 3],
    "leg_room_service": [5, 3, 4, 2],
    "baggage_handling": [4, 2, 5, 3],
    "checkin_service": [5, 3, 4, 2],
    "inflight_service": [4, 3, 5, 2],
    "cleanliness": [5, 3, 4, 2],
    "departure_delay_in_minutes": [10, 0, 15, 5],
    "arrival_delay_in_minutes": [5.0, 0.0, 3.5, 7.5],
    "satisfaction": ["satisfied", "neutral or dissatisfied", "satisfied", "neutral or dissatisfied"]
})


invalid_sample_data = pd.DataFrame({
    "gender": ["Male", "Female", "Unknown", "Male"],
    "customer_type": ["Loyal Customer", "Disloyal Customer", "Loyal Customer", "Loyal Customer"],
    "age": [200, -5, 50, 30],
    "type_of_travel": ["Business travel", "Personal Travel", "Leisure", "Business travel"],
    "class": ["Eco", "Eco Plus", "Luxury", "Business"],
    "flight_distance": [-100, 100000, 500, 2000],
    "inflight_wifi_service": [6, 10, 0, 4], 
    "time_convenient": [-1, 7, 3, 6],  
    "ease_of_online_booking": [10, 3, -2, 7],  
    "gate_location": [-5, 10, 2, 3], 
    "food_and_drink": [6, 8, 0, 9],  
    "online_boarding": [9, 10, 1, 6],  
    "seat_comfort": [10, 9, 1, 8],  
    "inflight_entertainment": [10, 9, 0, 7],  
    "on_board_service": [9, 10, 2, 8],  
    "leg_room_service": [10, 9, 0, 7],  
    "baggage_handling": [10, 9, 1, 8],  
    "checkin_service": [9, 10, 2, 8],  
    "inflight_service": [9, 10, 2, 8],  
    "cleanliness": [10, 9, 1, 8],  
    "departure_delay_in_minutes": [-10, 1000, 30, 0],  
    "arrival_delay_in_minutes": [-10.0, 100.0, -5.0, 20.0],  
    "satisfaction": ["neutral or dissatisfied", "satisfied", "angry", "satisfied"],
})


# Sample Train Data
sample_train_data = pd.DataFrame({
    "gender": ["Male", "Female", "Male", "Female"],
    "customer_type": ["Loyal Customer", "Disloyal Customer", "Loyal Customer", "Disloyal Customer"],
    "age": [25, 45, 35, 50],
    "type_of_travel": ["Business travel", "Personal Travel", "Personal Travel", "Business travel"],
    "class": ["Business", "Eco", "Eco Plus", "Business"],
    "flight_distance": [1500, 2000, 500, 2500],
    "inflight_wifi_service": [4, 3, 5, 2],
    "time_convenient": [4, 2, 5, 3],
    "ease_of_online_booking": [5, 3, 4, 2],
    "gate_location": [3, 4, 2, 5],
    "food_and_drink": [4, 5, 3, 2],
    "online_boarding": [4, 2, 5, 3],
    "seat_comfort": [5, 3, 4, 2],
    "inflight_entertainment": [4, 3, 5, 2],
    "on_board_service": [4, 2, 5, 3],
    "leg_room_service": [5, 3, 4, 2],
    "baggage_handling": [4, 2, 5, 3],
    "checkin_service": [5, 3, 4, 2],
    "inflight_service": [4, 3, 5, 2],
    "cleanliness": [5, 3, 4, 2],
    "departure_delay_in_minutes": [10, 0, 15, 5],
    "arrival_delay_in_minutes": [5.0, 0.0, 3.5, 7.5],
    "satisfaction": ["satisfied", "neutral or dissatisfied", "satisfied", "neutral or dissatisfied"]
})

# Sample Test Data
sample_test_data = pd.DataFrame({
    "gender": ["Female", "Male", "Female", "Male"],
    "customer_type": ["Loyal Customer", "Disloyal Customer", "Loyal Customer", "Disloyal Customer"],
    "age": [30, 40, 60, 29],
    "type_of_travel": ["Personal Travel", "Business travel", "Personal Travel", "Business travel"],
    "class": ["Eco", "Business", "Eco Plus", "Eco"],
    "flight_distance": [1200, 2200, 800, 1800],
    "inflight_wifi_service": [3, 4, 2, 5],
    "time_convenient": [5, 3, 4, 4],
    "ease_of_online_booking": [4, 5, 3, 2],
    "gate_location": [4, 2, 5, 3],
    "food_and_drink": [5, 4, 3, 2],
    "online_boarding": [3, 4, 2, 5],
    "seat_comfort": [4, 5, 3, 2],
    "inflight_entertainment": [5, 3, 4, 2],
    "on_board_service": [3, 4, 5, 2],
    "leg_room_service": [4, 3, 5, 2],
    "baggage_handling": [2, 3, 4, 5],
    "checkin_service": [3, 5, 2, 4],
    "inflight_service": [4, 5, 3, 2],
    "cleanliness": [5, 4, 3, 2],
    "departure_delay_in_minutes": [8, 3, 18, 4],
    "arrival_delay_in_minutes": [4.0, 1.5, 5.5, 6.0],
    "satisfaction": ["neutral or dissatisfied", "satisfied", "satisfied", "neutral or dissatisfied"]
})


