API_BASE_URL = ""

ALERT_THRESHOLDS = {
    "IMMINENT_HAZARD_STATUS": "IMMINENT_HAZARD",
    "ETA_MINUTES_MAX": 35,
    "SEVERE_LEVELS": ["CRITICAL", "HIGH"]
}

REGIONS = {
    "delhi_ncr": {
        "id": "delhi_ncr",
        "name": "Delhi-NCR & Northern Plains",
        "radar": "IMD Palam & Mausam Bhavan C-Band DWR",
        "bounds": [[27.8, 76.2], [29.3, 78.2]],
        "center": [28.55, 77.2],
        "has_real_data": True,
        "locations": [
            {"name": "Delhi IGI Airport (Aviation Hub)", "type": "airport", "lat": 28.556, "lng": 77.100, "color": "#ef4444", "baseEta": 18, "vil": 38, "dbz": 52, "gust": 64},
            {"name": "Gurugram CyberCity (Tech Hub)",     "type": "city",    "lat": 28.495, "lng": 77.089, "color": "#f59e0b", "baseEta": 32, "vil": 24, "dbz": 46, "gust": 50},
            {"name": "Noida Sector 62 (Industrial Zone)", "type": "city",    "lat": 28.628, "lng": 77.365, "color": "#38bdf8", "baseEta": -10, "vil": 5, "dbz": 20, "gust": 15},
            {"name": "Faridabad Agri-Belt (Rural)",       "type": "agri",    "lat": 28.408, "lng": 77.317, "color": "#10b981", "baseEta": -60, "vil": 2, "dbz": 15, "gust": 10}
        ]
    },
    "himalayan_belt": {
        "id": "himalayan_belt",
        "name": "Himalayan Belt",
        "radar": "IMD via RainViewer",
        "bounds": [[29.8, 77.6], [31.0, 79.6]],
        "center": [30.4, 78.6],
        "has_real_data": True,
        "locations": [
            {"name": "Dehradun Airport", "type": "airport", "lat": 30.189, "lng": 78.180, "color": "#ef4444", "baseEta": 45, "vil": 0, "dbz": 0, "gust": 0},
            {"name": "Shimla Tourist Hub", "type": "city", "lat": 31.104, "lng": 77.173, "color": "#f59e0b", "baseEta": 60, "vil": 0, "dbz": 0, "gust": 0},
            {"name": "Uttarkashi Agri", "type": "agri", "lat": 30.726, "lng": 78.435, "color": "#10b981", "baseEta": 120, "vil": 0, "dbz": 0, "gust": 0}
        ]
    },
    "mumbai_ghats": {
        "id": "mumbai_ghats",
        "name": "Mumbai & Western Ghats",
        "radar": "IMD via RainViewer",
        "bounds": [[18.5, 72.3], [19.7, 73.6]],
        "center": [19.05, 72.9],
        "has_real_data": True,
        "locations": [
            {"name": "CSIA Mumbai", "type": "airport", "lat": 19.089, "lng": 72.865, "color": "#ef4444", "baseEta": 15, "vil": 0, "dbz": 0, "gust": 0},
            {"name": "Navi Mumbai", "type": "city", "lat": 19.033, "lng": 73.029, "color": "#f59e0b", "baseEta": 25, "vil": 0, "dbz": 0, "gust": 0},
            {"name": "Lonavala Ghats", "type": "agri", "lat": 18.748, "lng": 73.405, "color": "#10b981", "baseEta": -20, "vil": 0, "dbz": 0, "gust": 0}
        ]
    },
    "northeast_bengal": {
        "id": "northeast_bengal",
        "name": "Northeast & Meghalaya",
        "radar": "IMD via RainViewer",
        "bounds": [[25.0, 91.0], [26.4, 92.6]],
        "center": [25.8, 91.8],
        "has_real_data": True,
        "locations": [
            {"name": "Guwahati Airport", "type": "airport", "lat": 26.106, "lng": 91.585, "color": "#ef4444", "baseEta": -5, "vil": 0, "dbz": 0, "gust": 0},
            {"name": "Shillong City", "type": "city", "lat": 25.578, "lng": 91.893, "color": "#f59e0b", "baseEta": -15, "vil": 0, "dbz": 0, "gust": 0},
            {"name": "Cherrapunji Agri", "type": "agri", "lat": 25.270, "lng": 91.732, "color": "#10b981", "baseEta": 10, "vil": 0, "dbz": 0, "gust": 0}
        ]
    }
}
