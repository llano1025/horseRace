import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from fake_useragent import UserAgent


def get_race_dates():
    # URL of the HKJC race results page
    url = 'https://racing.hkjc.com/racing/information/English/racing/LocalResults.aspx'

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the <select> tag with id 'selectId'
        select_tag = soup.find('select', id='selectId')

        # Extract the values of all <option> tags within the <select> tag
        dates = [option['value'] for option in select_tag.find_all('option')]

        # Print the extracted dates
        for date in dates:
            print(date)
            return dates
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        return None


def get_race_card(race_date):
    # Find the maximum race number
    max_race_no = get_race_number(race_date, False)

    # Initialize a list to store race card information
    race_card = []

    for race_no in range(1, max_race_no + 1):
        # URL of the HKJC race card page
        url = f'https://racing.hkjc.com/racing/information/English/Racing/RaceCard.aspx?RaceDate={race_date}&RaceNo={race_no}'

        # Send a GET request to the page and get the HTML content
        ua = UserAgent(browsers=['edge', 'chrome', 'firefox'])
        user_agent = ua.random
        headers = {'user-agent': user_agent}

        # Send a GET request to the URL
        response = requests.get(url, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract the race name
            race_name = soup.find('span', class_='font_wb').text.strip()
            remaining_text = soup.find('div', class_='f_fs13').decode_contents().split('<br>')
            date_location_time = remaining_text[1].strip()
            race_date, race_location, race_time = [item.strip() for item in date_location_time.split(',')]
            course_details = remaining_text[2].strip()
            course_type, course_details = course_details.split(',')
            course, distance, condition = [item.strip() for item in course_details.split(',')]
            prize_and_type = remaining_text[3].strip().replace('Prize Money: ', '')
            prize_money, _, race_type = [item.strip() for item in prize_and_type.split(',')]

            # Create a dictionary with the extracted data
            race_data = {
                "Race Name": race_name,
                "Race Date": race_date,
                "Race Location": race_location,
                "Race Time": race_time,
                "Course Type": course_type.strip(),
                "Course": course.strip(),
                "Distance": distance.strip(),
                "Condition": condition.strip(),
                "Race Type": race_type
            }

            # Find the table containing the race card data
            race_table = soup.find('table', id='racecardlist')

            # Extract data from each row of the table
            for row in race_table.find_all('tr')[3:]:  # Skip the header row
                cols = row.find_all('td')
                horse_info = {
                    'number': cols[0].text.strip(),
                    'horse_name': cols[3].text.strip(),
                    'brand_number': cols[4].text.strip(),
                    'weight': cols[5].text.strip(),
                    'jockey': cols[6].text.strip(),
                    'draw': cols[8].text.strip(),
                    'trainer': cols[9].text.strip(),
                    'horse_weight': cols[13].text.strip(),
                    'horse_age': cols[16].text.strip(),
                    'horse_gender': cols[18].text.strip(),
                    'season_stake': cols[19].text.strip(),
                    'days_last_run': cols[21].text.strip(),
                    'gear': cols[22].text.strip()
                }
                horse_info.update(race_data)
                race_card.append(horse_info)
        else:
            print(f"Failed to retrieve the page. Status code: {response.status_code}")
            return None

    # Convert race card list to a DataFrame
    df = pd.DataFrame(race_card)

    # Output the data
    return df


def get_race_number(race_date, IS_RACE_RESULT):
    # Define the URL of the race results page
    if IS_RACE_RESULT:
        url = f'https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate={race_date}&RaceNo=1'
    else:
        url = f'https://racing.hkjc.com/racing/information/English/racing/RaceCard.aspx'

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all img tags with the src attribute containing 'racecard_rt_'
        img_tags = soup.find_all('img', src=re.compile(r'racecard_rt_\d+\.gif'))

        # Extract the race numbers from the img src attributes
        race_numbers = [int(re.search(r'racecard_rt_(\d+)\.gif', img['src']).group(1)) for img in img_tags]

        # Find the maximum race number
        max_race_number = max(race_numbers) if race_numbers else None

        return max_race_number


def get_race_result(race_date):
    rows = []
    headers = []
    max_race_no = get_race_number(race_date, True)

    for race_no in range(1, max_race_no + 1):
        # Define the URL of the race results page
        url = f'https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate={race_date}&RaceNo={race_no}'

        # Send a GET request to the page and get the HTML content
        ua = UserAgent(browsers=['edge', 'chrome', 'firefox'])
        user_agent = ua.random
        headers = {'user-agent': user_agent}

        # Send a GET request to the URL
        response = requests.get(url, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the table containing the race results
            table = soup.find('table', class_='f_tac table_bd draggable')

            # Find the race information
            text_str1 = soup.find('span', class_='f_fl f_fs13').text
            span_text = text_str1.replace('Race Meeting:', '').strip()
            race_date, race_location = span_text.split('  ')[0].strip(), span_text.split('  ')[-1].strip()

            text_str2 = soup.find('tbody', class_='f_fs13').find_all('tr')
            race_class_distance = text_str2[1].find_all('td')[0].text.strip()
            going = text_str2[1].find_all('td')[2].text.strip()
            course = text_str2[2].find_all('td')[2].text.strip()

            # Find the header row
            for row in table.find_all('tr')[0:1]:  # Skip the header row
                headers = row.find_all('td')
                headers = [header.text.strip() for header in headers]
                headers.extend(['race_date', 'race_location', 'race_class_distance', 'going', 'course'])

            # Extract table rows
            for row in table.find_all('tr')[1:]:  # Skip the header row
                cols = row.find_all('td')
                cols = [col.text.strip() for col in cols]
                cols.extend([race_date, race_location, race_class_distance, going, course])
                rows.append(cols)

        else:
            print(f"Failed to retrieve the page. Status code: {response.status_code}")

    # Create a DataFrame
    df = pd.DataFrame(rows, columns=headers)

    # Print the DataFrame
    return df


race_dates = get_race_dates()
race_card = get_race_card(race_dates[0])
race_result = get_race_result(race_dates[1])

print(race_result)
