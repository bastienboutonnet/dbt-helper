a = {
    0: ["flight.flight_search"],
    1: [
        "predictive_modelling.base_flight_search_booking",
        ["kafka_topics.flight_search_event"],
        ["predictive_modelling.base_flight_search_result_air_fare"],
        ["predictive_modelling.base_flight_search_result"],
    ],
    2: [
        "kafka_topics.flight_booking_event",
        ["kafka_topics.flight_booking_event_traveler"],
        ["kafka_topics.flight_booking_event_flight_leg"],
        ["kafka_topics.flight_booking_event_flight_leg_segment"],
        ["kafka_topics.flight_search_event_result_air_fare"],
        ["kafka_topics.flight_search_event_result_leg"],
        ["kafka_topics.flight_search_event_result_deduped"],
    ],
}

