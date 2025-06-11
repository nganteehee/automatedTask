import os, random
capitals = {'Alabama': 'Montgomery', 'Alaska': 'Juneau', 'Arizona': 'Phoenix', 
'Arkansas': 'Little Rock', 'California': 'Sacramento', 'Colorado': 'Denver', 
'Connecticut': 'Hartford', 'Delaware': 'Dover', 'Florida': 'Tallahassee', 
'Georgia': 'Atlanta', 'Hawaii': 'Honolulu', 'Idaho': 'Boise', 'Illinois': 
'Springfield', 'Indiana': 'Indianapolis', 'Iowa': 'Des Moines', 'Kansas': 
'Topeka', 'Kentucky': 'Frankfort', 'Louisiana': 'Baton Rouge', 'Maine': 
'Augusta', 'Maryland': 'Annapolis', 'Massachusetts': 'Boston', 'Michigan': 
'Lansing', 'Minnesota': 'Saint Paul', 'Mississippi': 'Jackson', 'Missouri': 
'Jefferson City', 'Montana': 'Helena', 'Nebraska': 'Lincoln', 'Nevada': 
'Carson City', 'New Hampshire': 'Concord', 'New Jersey': 'Trenton', 'New Mexico': 'Santa Fe', 'New York': 'Albany', 'North Carolina': 'Raleigh', 
'North Dakota': 'Bismarck', 'Ohio': 'Columbus', 'Oklahoma': 'Oklahoma City', 
'Oregon': 'Salem', 'Pennsylvania': 'Harrisburg', 'Rhode Island': 'Providence', 
'South Carolina': 'Columbia', 'South Dakota': 'Pierre', 'Tennessee': 
'Nashville', 'Texas': 'Austin', 'Utah': 'Salt Lake City', 'Vermont': 
'Montpelier', 'Virginia': 'Richmond', 'Washington': 'Olympia', 'West Virginia': 'Charleston', 'Wisconsin': 'Madison', 'Wyoming': 'Cheyenne'}
states = list(capitals.keys())
for quiztest in range(4):
    random.shuffle(states) 
    with open(f'capitalsquiz %s.txt' %(quiztest + 1), 'w') as testFile:
        testFile.write('Name: \n\n')
        testFile.write('Date: \n\n')
        testFile.write('Period: \n\n')
        testFile.write('Stata Capitals Quiz Form (%s) \n' % (quiztest+1))
        for i in range(1, 50):
            list_capital = list(capitals.values())
            correctAnswer = capitals[states[i]]
            del list_capital[list_capital.index(correctAnswer)]
            wrong_answer = random.sample(list_capital, 3)
            list_answer = [correctAnswer] + wrong_answer
            testFile.write('%s What is the capital of %s \n' %(i, states[i]))
            for j in range(4):
                testFile.write('ABCD'[j] + ' ' + list_answer[j] + '\n')
    with open(f'Answer key %s' %(quiztest+1), 'w') as answer_key:
        for i in range(1, 50):
            answer_key.write('%s %s' %(i, capitals[states[i]]))

    