import requests
import sys
sys.path.append('src')
from app.config import settings

# Test CFBD API directly
api_key = settings.cfbd_api_key
print(f'Testing CFBD API with key: {api_key[:10]}...')

headers = {
    'Authorization': f'Bearer {api_key}',
    'Accept': 'application/json'
}

url = 'https://api.collegefootballdata.com/teams'
response = requests.get(url, headers=headers)

print(f'Status Code: {response.status_code}')
if response.status_code == 200:
    teams = response.json()
    print(f'Successfully fetched {len(teams)} teams')
    
    # Look for Tennessee
    tennessee_teams = [team for team in teams if 'tennessee' in team.get('name', '').lower()]
    print('Tennessee teams found:')
    for team in tennessee_teams:
        print(f'  - {team["name"]} ({team.get("conference", "Unknown")})')
else:
    print(f'Error: {response.text}')
