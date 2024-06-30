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
import numpy as np


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
    try:
        # Find the maximum race number
        max_race_no = get_race_number(race_date, False)

        # Initialize a list to store race card information
        race_card = []
        date_obj = datetime.strptime(race_date, '%d/%m/%Y')
        format_date = date_obj.strftime('%Y/%m/%d')

        for race_no in range(1, max_race_no):
            # URL of the HKJC race card page
            url = f'https://racing.hkjc.com/racing/information/English/Racing/RaceCard.aspx?RaceDate={format_date}&RaceNo={race_no}'

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
                remaining_text = remaining_text[0].split('<br/>')
                match = re.search(r'Race (\d+)', remaining_text[0])
                if match:
                    race_name = int(match.group(1))
                else:
                    race_name = ''
                race_location = remaining_text[1].split(',')[-2].strip()
                race_time = remaining_text[1].split(',')[-1].strip()
                course_details = remaining_text[2].strip()
                course = ' - '.join(course_details.split(',')[0:-2])
                distance = course_details.split(',')[-2].strip()
                match = re.search(r'(\d+)M', distance)
                if match:
                    distance = int(match.group(1))
                else:
                    distance = ''
                condition = course_details.split(',')[-1].strip()
                prize_and_type = remaining_text[3].strip().replace('Prize Money: ', '')
                prize_money, _, race_type = [item.strip() for item in prize_and_type.split(', ')]

                # Create a dictionary with the extracted data
                race_data = {
                    "Race Name": race_name,
                    "Race Date": race_date,
                    "Race Location": race_location,
                    "Race Time": race_time,
                    "Course": course.strip(),
                    "Distance": distance,
                    "Going": condition.strip(),
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

    except Exception:
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
            _class = race_class_distance.split(' - ')[0]
            distance = race_class_distance.split(' - ')[1]
            match = re.search(r'(\d+)M', distance)
            if match:
                distance = int(match.group(1))
            else:
                distance = ''
            going = text_str2[1].find_all('td')[2].text.strip()
            course = text_str2[2].find_all('td')[2].text.strip()

            # Find the header row
            for row in table.find_all('tr')[0:1]:  # Skip the header row
                headers = row.find_all('td')
                headers = [header.text.strip() for header in headers]
                headers.extend(['Race Date', 'Race Location', 'Class', 'Distance', 'Going', 'Course'])

            # Extract table rows
            for row in table.find_all('tr')[1:]:  # Skip the header row
                cols = row.find_all('td')
                cols = [col.text.strip() for col in cols]
                cols.extend([race_date, race_location, _class, distance, going, course])
                rows.append(cols)

            # Introduce a time delay between requests
            time_delay = random.uniform(2, 5)  # Random delay between 2 and 5 seconds
            print(f"Sleeping for {time_delay:.2f} seconds")
            time.sleep(time_delay)

        else:
            print(f"Failed to retrieve the page. Status code: {response.status_code}")

    # Create a DataFrame
    df = pd.DataFrame(rows, columns=headers)
    df[['Horse', 'Brand No.']] = df['Horse'].str.extract(r'(.+?)\xa0\((.+?)\)')
    columns = list(df.columns)
    last_column = columns.pop()
    columns.insert(3, last_column)
    df = df[columns]

    # Print the DataFrame
    return df


def get_horse_details():
    # Extract unique 'Brand No.' values from past race result
    latest_result = get_latest_csv('./data/pastRaceResult/')
    df = pd.read_csv(os.path.join('./data/pastRaceResult/', latest_result))
    brand_numbers_result = df['Brand No.'].unique()

    #  Extract unique "Brand No." values from horse information
    latest_horseinfo = get_latest_csv('./data/horseInformation/')
    if latest_horseinfo:
        df = pd.read_csv(os.path.join('./data/horseInformation/', latest_horseinfo))
        brand_numbers_info = df['Brand No.'].unique()
        brand_numbers = set(brand_numbers_result) - set(brand_numbers_info)
        IS_MERGE_REQ = True
    else:
        brand_numbers = brand_numbers_result
        IS_MERGE_REQ = False

    # Generate years from the current year to 2018
    current_year = datetime.now().year
    years = list(range(current_year, 2017, -1))
    df_list = []

    for brand_no in brand_numbers:
        for year in years:
            url = f"https://racing.hkjc.com/racing/information/english/Horse/OtherHorse.aspx?HorseId=HK_{str(year)}_{brand_no}"

            # Send a GET request to the page and get the HTML content
            ua = UserAgent(browsers=['edge', 'chrome', 'firefox'])
            user_agent = ua.random
            headers = {'user-agent': user_agent}

            # Send a GET request to the URL
            response = requests.get(url, headers=headers)

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the HTML content using BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')

                # Locate the table with class 'horseProfile'
                table = soup.find('table', class_='horseProfile')

                if not table:
                    print("Horse profile table not found.")
                    continue  # Exit the inner loop if table is not found

                horse_data = {}

                # Extract the horse name and ID from the title
                title = table.find('span', class_='title_text')
                if title:
                    horse_data['Horse Name and ID'] = title.text.strip()
                    print(horse_data['Horse Name and ID'])

                # Extract data from the first column (left side)
                first_col = table.find_all('td', style='width: 280px;')[0]
                horse_data['Image URL'] = first_col.find('img')['src'] if first_col.find('img') else None

                # Extract data from the second column (middle)
                second_col = table.find_all('td', valign='top', style='width: 260px;')[0]
                rows = second_col.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) == 3:
                        key = cells[0].text.strip()
                        value = cells[2].text.strip()
                        horse_data[key] = value

                # Extract data from the third column (right side)
                third_col = table.find_all('td', valign='top', style='width: 280px;')[0]
                rows = third_col.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) == 3:
                        key = cells[0].text.strip()
                        value = cells[2].text.strip()
                        horse_data[key] = value

                df_list.append(pd.DataFrame([horse_data]))
                if horse_data:
                    break  # Exit the inner loop if data is found and appended

    horse_df = pd.concat(df_list)
    horse_df = reformat_horse_info(horse_df)
    save_horse_info(horse_df, IS_MERGE_REQ)
    return horse_df


def reformat_time(df):
    # Parse the original strings and convert to datetime format
    df = df.dropna(subset=['Pla.'])
    df = df.drop(df[df['Pla.'] == 'VOID'].index)
    df['Race Date'] = pd.to_datetime(df['Race Date'], format='%d/%m/%Y')

    # Format the datetime objects as strings in the desired format
    df['Race Date'] = df['Race Date'].dt.strftime('%Y-%m-%d')
    return df


def reformat_horse_info(df):
    # Extract the parts with three sets of parentheses
    df[['Horse', 'Brand No.', 'Status']] = df['Horse Name and ID'].str.extract(r'(.+?)\s+\((.+?)\)(?:\s+\((.+?)\))?')
    df['Status'] = np.where(df['Horse Name and ID'].str.contains(r'\((.+?)\)\s+\((.+?)\)'), df['Status'], np.nan)
    try:
        df[['Country of Origin (1)', 'Age']] = df['Country of Origin / Age'].str.split(' / ', expand=True)
        df.rename(columns={'Country of Origin': 'Country of Origin (2)'}, inplace=True)
        df['Country of Origin'] = df['Country of Origin (1)'].fillna(df['Country of Origin (2)'])
        df = df.drop(columns=['Country of Origin (1)', 'Country of Origin (2)'])
    except KeyError as exception:
        print(exception)
    df[['Color', 'Sex']] = df['Colour / Sex'].str.split(' / ', 1, expand=True)
    df['Sex'] = df['Sex'].apply(lambda x: x.split(' / ')[-1] if ' / ' in x else x)
    return df


def save_dataframe_to_csv(dataframe, path, IS_MERGE_REQ):
    try:
        #  Check if merge with existing data is required
        if IS_MERGE_REQ:
            latest_file = get_latest_csv(path)
            dataframe_master = pd.read_csv(os.path.join(path, latest_file))
            dataframe = pd.concat([dataframe, dataframe_master])

        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dataframe = reformat_time(dataframe)

        # Save the DataFrame to a CSV file
        current_date = datetime.now().strftime("%Y.%m.%d")
        dataframe.to_csv(os.path.join(path, f'{current_date}_race_result.csv'), index=False)
        print(f"DataFrame successfully saved to {path}")

    except Exception as e:
        print(f"An error occurred while saving the DataFrame: {e}")


def save_horse_info(horse_df, IS_MERGE_REQ):
    path = './data/horseInformation/'

    #  Load past horse information
    if IS_MERGE_REQ:
        latest_file = get_latest_csv(path)
        dataframe_master = pd.read_csv(os.path.join(path, latest_file))
        horse_df = pd.concat([horse_df, dataframe_master])

    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save the DataFrame to a CSV file
    current_date = datetime.now().strftime("%Y.%m.%d")
    horse_df.to_csv(
        os.path.join(path, f'{current_date}_horse_info.csv'),
        index=False)
    print(f"DataFrame successfully saved to {path}")


def get_latest_csv(directory):
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    if not csv_files:
        return None

    date_files = {}
    for file in csv_files:
        match = re.match(r'(\d{4}\.\d{2}\.\d{2})_(race_result|horse_info)\.csv', file)
        if match:
            date_str = match.group(1)
            date_obj = datetime.strptime(date_str, '%Y.%m.%d')
            date_files[date_obj] = file

    if date_files:
        latest_date = max(date_files.keys())
        return date_files[latest_date]
    return None


def filename_to_date(filename):
    match = re.match(r'(\d{4}\.\d{2}\.\d{2})_race_result\.csv', filename)
    if match:
        date_str = match.group(1)
        return datetime.strptime(date_str, '%Y.%m.%d')
    return None


def filter_dates(dates_list, comparison_date):
    filtered_dates = [date for date in dates_list if
                      datetime.now() > datetime.strptime(date.split("\r")[0], '%d/%m/%Y') > comparison_date]
    return filtered_dates


def get_updated_dates(directory, dates_list):
    latest_file = get_latest_csv(directory)
    if latest_file:
        latest_date = filename_to_date(latest_file)
        print(f"Latest file: {latest_file}")
        print(f"Latest date: {latest_date}")

        filtered_dates = filter_dates(dates_list, latest_date)
        print(f"Dates greater than {latest_date}: {filtered_dates}")
        return filtered_dates, True
    else:
        print("No CSV files found in the directory.")
        return dates_list, False


def save_past_race_result():
    # Programme initiation
    path = './data/pastRaceResult/'

    # Get the past race result for race dates
    race_dates = get_race_dates()
    race_dates, IS_MERGE_REQ = get_updated_dates(path, race_dates)
    race_result_list = []
    for race_date in race_dates:
        try:
            race_result = get_race_result(race_date)
        except Exception as exception:
            print(exception)
            continue
        race_result_list.append(race_result)
    if race_result_list:
        race_results = pd.concat(race_result_list)
    else:
        race_results = None

    # Save the result into .csv
    save_dataframe_to_csv(race_results, path, IS_MERGE_REQ)


def get_future_race_card():
    # Get the past race result for race dates
    race_dates = get_race_dates()
    future_dates = [date for date in race_dates if
                    datetime.now() < datetime.strptime(date.split("\r")[0], '%d/%m/%Y')]
    for date in future_dates:
        dataframe = get_race_card(date)
        return dataframe
