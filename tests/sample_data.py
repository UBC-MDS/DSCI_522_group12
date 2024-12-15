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
