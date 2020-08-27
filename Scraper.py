from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import re
import xlsxwriter

def scrape_prepositions():
    driver = webdriver.Chrome()
    driver.get("https://www.englishclub.com/grammar/prepositions-list.htm")
    all_ul = driver.find_elements_by_xpath("//main/ul")
    prepositions = ''
    for item in all_ul:
        prepositions += item.text.rstrip().lower() + ' '

    prepositions_arr = prepositions.split()
    # print(prepositions_arr)

    prepositions_arr.insert(0, 'Prepositions')
    driver.close()
    return prepositions_arr

def scrape_sports():
    driver = webdriver.Chrome()
    driver.get("https://www.topendsports.com/sport/list/index.htm")
    all_ul = driver.find_elements_by_xpath("//div/div/ul/li/a[1]")
    sports = []
    for item in all_ul:
        sports.append(item.text.rstrip().lower())

    # eliminate the random hyperlinks at the end
    reduce_array = sports[:-15]
    # print(reduce_array)
    
    reduce_array.insert(0, 'Sports')
    driver.close()
    return reduce_array

def scrape_politics():
    driver = webdriver.Chrome()
    driver.get("https://myvocabulary.com/word-list/politics-vocabulary/")
    all_ul = driver.find_elements_by_xpath("//tbody/tr/td[2]")
    politics = []
    for item in all_ul:
        temp_arr = item.text.rstrip().split(' ')
        for word in temp_arr:
            regex = re.compile('[^a-zA-Z]')
            word = regex.sub('', word)
            politics.append(word.strip().lower())

    politics.insert(0, 'Politics')
    driver.close()
    return politics

def scrape_social_issues():
    driver = webdriver.Chrome()
    driver.get("https://www.isidewith.com/polls/popular")
    all_ul = driver.find_elements_by_xpath("//div/div/a/div/p/span")
    social = []
    for item in all_ul:
        social.append(item.text.lower())
    
    social.insert(0, "Social Justice")
    social.append('blm')
    social.append('BLM')
    driver.close()
    return social

def scrape_science():
    driver = webdriver.Chrome()
    driver.get("https://reversedictionary.org/wordsfor/science")
    all_ul = driver.find_elements_by_xpath("//div/div/div[2]/div[3]/a")
    science = []
    for item in all_ul:
        science.append(item.text.lower())
    
    science.insert(0, "Science")
    driver.close()
    return science


names = []

def scrape_names(names):
    driver = webdriver.Chrome()
    driver.get("https://family.disney.com/articles/1000-most-popular-boy-names/")
    all_ul = driver.find_elements_by_xpath("//body/div/div/article/div/main/ol/li")
    for item in all_ul:
        if len(names) > 450:
            break
        names.append(item.text.lower())
    
    names.insert(0, "Names")
    driver.close()

def scrape_names_two(names):
    driver = webdriver.Chrome()
    driver.get("https://www.whattoexpect.com/baby-names/list/top-baby-names-for-girls/")
    girls_ul = driver.find_elements_by_xpath("//body/div/div/div[2]/div[2]/article/div[2]/section[2]/div[2]/div/div[2]/ol/li")
    for item in girls_ul:
        if len(names) > 900:
            break
        names.append(item.text.lower())
    driver.close()
    
def create_spreadsheet():
    workbook = xlsxwriter.Workbook('Words.xlsx')
    worksheet = workbook.add_worksheet()

    scrape_names(names)
    scrape_names_two(names)

    array = [scrape_prepositions(),
            scrape_sports(),
            scrape_social_issues(),
            scrape_politics(),
            scrape_science(),
            names]

    row = 0

    for col, data in enumerate(array):
        worksheet.write_column(row, col, data)

    workbook.close()

create_spreadsheet()
