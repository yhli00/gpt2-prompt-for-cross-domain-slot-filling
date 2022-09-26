domain2slots = {
    "AddToPlaylist": ['music_item', 'playlist_owner', 'entity_name', 'playlist', 'artist'],
    "BookRestaurant": ['city', 'facility', 'timeRange', 'restaurant_name', 'country', 'cuisine', 'restaurant_type', 'served_dish', 'party_size_number', 'poi', 'sort', 'spatial_relation', 'state', 'party_size_description'],
    "GetWeather": ['city', 'state', 'timeRange', 'current_location', 'country', 'spatial_relation', 'geographic_poi', 'condition_temperature', 'condition_description'],
    "PlayMusic": ['genre', 'music_item', 'service', 'year', 'playlist', 'album','sort', 'track', 'artist'],
    "RateBook": ['object_part_of_series_type', 'object_select', 'rating_value', 'object_name', 'object_type', 'rating_unit', 'best_rating'],
    "SearchCreativeWork": ['object_name', 'object_type'],
    "SearchScreeningEvent": ['timeRange', 'movie_type', 'object_location_type','object_type', 'location_name', 'spatial_relation', 'movie_name']
}

domain_set = ["AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]

label_verb = {
  "PlayMusic": {
    "album": "album",
    "artist": "artist",
    "genre": "genre",
    "music_item": "music item",
    "playlist": "playlist",
    "service": "service",
    "sort": "sort",
    "track": "track",
    "year": "year"
  },
  "AddToPlaylist": {
    "artist": "artist",
    "entity_name": "entity name",
    "music_item": "music item",
    "playlist": "playlist",
    "playlist_owner": "playlist owner"
  },
  "RateBook": {
    "best_rating": "best rating",
    "object_name": "object name",
    "object_part_of_series_type": "object part of series type",
    "object_select": "object select",
    "object_type": "object type",
    "rating_unit": "rating unit",
    "rating_value": "rating value"
  },
  "SearchScreeningEvent": {
    "location_name": "location name",
    "movie_name": "movie name",
    "movie_type": "movie type",
    "object_location_type": "object location type",
    "object_type": "object type",
    "spatial_relation": "spatial relation",
    "timeRange": "time range"
  },
  "BookRestaurant": {
    "city": "city",
    "country": "country",
    "cuisine": "cuisine",
    "facility": "facility",
    "party_size_description": "party size description",
    "party_size_number": "party size number",
    "poi": "point of interest",
    "restaurant_name": "restaurant name",
    "restaurant_type": "restaurant type",
    "served_dish": "served dish",
    "sort": "sort",
    "spatial_relation": "spatial relation",
    "state": "state",
    "timeRange": "time range"
  },
  "SearchCreativeWork": {
    "object_name": "object name",
    "object_type": "object type"
  },
  "GetWeather": {
    "city": "city",
    "condition_description": "condition description",
    "condition_temperature": "condition temperature",
    "country": "country",
    "current_location": "current location",
    "geographic_poi": "geographic point of interest",
    "spatial_relation": "spatial relation",
    "state": "state",
    "timeRange": "time range"
  }
}


verb_label = {
  "PlayMusic": {
    "album": "album",
    "artist": "artist",
    "genre": "genre",
    "music item": "music_item",
    "playlist": "playlist",
    "service": "service",
    "sort": "sort",
    "track": "track",
    "year": "year"
  },
  "AddToPlaylist": {
    "artist": "artist",
    "entity name": "entity_name",
    "music item": "music_item",
    "playlist": "playlist",
    "playlist owner": "playlist_owner"
  },
  "RateBook": {
    "best rating": "best_rating",
    "object name": "object_name",
    "object part of series type": "object_part_of_series_type",
    "object select": "object_select",
    "object type": "object_type",
    "rating unit": "rating_unit",
    "rating value": "rating_value"
  },
  "SearchScreeningEvent": {
    "location name": "location_name",
    "movie name": "movie_name",
    "movie type": "movie_type",
    "object location type": "object_location_type",
    "object type": "object_type",
    "spatial relation": "spatial_relation",
    "time range": "timeRange"
  },
  "BookRestaurant": {
    "city": "city",
    "country": "country",
    "cuisine": "cuisine",
    "facility": "facility",
    "party size description": "party_size_description",
    "party size number": "party_size_number",
    "point of interest": "poi",
    "restaurant name": "restaurant_name",
    "restaurant type": "restaurant_type",
    "served dish": "served_dish",
    "sort": "sort",
    "spatial relation": "spatial_relation",
    "state": "state",
    "time range": "timeRange"
  },
  "SearchCreativeWork": {
    "object name": "object_name",
    "object type": "object_type"
  },
  "GetWeather": {
    "city": "city",
    "condition description": "condition_description",
    "condition temperature": "condition_temperature",
    "country": "country",
    "current location": "current_location",
    "geographic point of interest": "geographic_poi",
    "spatial relation": "spatial_relation",
    "state": "state",
    "time range": "timeRange"
  }
}
