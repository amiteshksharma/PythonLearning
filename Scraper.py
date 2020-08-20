from selenium import webdriver
from selenium.webdriver.common.keys import Keys

driver = webdriver.Chrome()
driver.get("https://www.englishclub.com/grammar/prepositions-list.htm")
all_ul = driver.find_elements_by_xpath("//main/ul")
prepositions = ''
for item in all_ul:
    prepositions += item.text.rstrip() + ' '

prepositions_arr = prepositions.split()
print(prepositions_arr)

driver.close()