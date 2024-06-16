import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from fake_useragent import UserAgent
import os
from datetime import datetime, timedelta
import sys
import pytz
import time
import random


def is_race_card_available():
    # Set the timezone to GMT+8
    gmt_plus_8 = pytz.timezone('Asia/Shanghai')  # 'Asia/Shanghai' is GMT+8
    now = datetime.now(gmt_plus_8)
    # Calculate the current week's Monday at noon
    monday_noon = (now - timedelta(days=now.weekday())).replace(hour=12, minute=00, second=00)
    # Calculate the current week's Wednesday end (midnight at the start of Thursday)
    wednesday_end = (monday_noon + timedelta(days=2)).replace(hour=23, minute=59, second=59)
    # Calculate the current week's Thursday at noon
    thursday_noon = (monday_noon + timedelta(days=3)).replace(hour=12, minute=00, second=00)
    # Calculate the current week's Saturday end (midnight at the start of Sunday)
    saturday_end = (thursday_noon + timedelta(days=2)).replace(hour=23, minute=59, second=59)
    # Check if 'now' is within Monday noon to Wednesday end
    if monday_noon <= now <= wednesday_end:
        return True
    # Check if 'now' is within Thursday noon to Saturday end
    if thursday_noon <= now <= saturday_end:
        return True

    return False


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
        return dates
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        return None


def get_race_card(race_date):

    if is_race_card_available():
        # Find the maximum race number
        max_race_no = get_race_number(race_date, False)

        # Initialize a list to store race card information
        race_card = []

        for race_no in range(1, max_race_no):
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
                        'Horse No.': cols[0].text.strip(),
                        'Horse': cols[3].text.strip(),
                        'Brand No.': cols[4].text.strip(),
                        'Act. Wt.': cols[5].text.strip(),
                        'Jockey': cols[6].text.strip(),
                        'Dr.': cols[8].text.strip(),
                        'Trainer': cols[9].text.strip(),
                        'Declar. Horse Wt.': cols[13].text.strip(),
                        'Age': cols[16].text.strip(),
                        'Sex': cols[18].text.strip(),
                        'Season Stake': cols[19].text.strip(),
                        'Days since Last Run': cols[21].text.strip(),
                        'Gear': cols[22].text.strip()
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

    else:
        print("Race Card not available")
        sys.exit()


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
    print(race_date)

    for race_no in range(1, max_race_no):
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
                headers.extend(['Race Date', 'Race Location', 'Race Class Distance', 'Going', 'Course'])

            # Extract table rows
            for row in table.find_all('tr')[1:]:  # Skip the header row
                cols = row.find_all('td')
                cols = [col.text.strip() for col in cols]
                cols.extend([race_date, race_location, race_class_distance, going, course])
                rows.append(cols)

            # Introduce a time delay between requests
            time_delay = random.uniform(2, 5)  # Random delay between 2 and 5 seconds
            print(f"Sleeping for {time_delay:.2f} seconds")
            time.sleep(time_delay)

        else:
            print(f"Failed to retrieve the page. Status code: {response.status_code}")

    # Create a DataFrame
    df = pd.DataFrame(rows, columns=headers)

    # Print the DataFrame
    return df


def save_dataframe_to_csv(dataframe, path):
    try:
        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save the DataFrame to a CSV file
        dataframe.to_csv(os.path.join(path, 'race_result.csv'), index=False)
        print(f"DataFrame successfully saved to {path}")
    except Exception as e:
        print(f"An error occurred while saving the DataFrame: {e}")


# Programme initiation
path = './pastRaceResult/'
race_dates = get_race_dates()

# Get the past race result for race dates
race_result_list = []
for race_date in race_dates:
    try:
        race_result = get_race_result(race_date)
    except Exception as exception:
        print(exception)
        continue
    race_result_list.append(race_result)
race_results = pd.concat(race_result_list)

# Save the result into .csv
save_dataframe_to_csv(race_results, path)
