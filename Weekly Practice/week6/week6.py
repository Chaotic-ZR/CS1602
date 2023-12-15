# 1
def dele(lst):
    a = set(lst)
    a = list(a)
    a = a[::-1]
    return list(a)

b = [1, 3, 3, 0, 1, 2, 5, 0]
print(dele(b))




# 2
def cap(*argv):
    a = [arg.upper() for arg in argv]
    for i in a:
        print(i)
        

cap("hello, world", "test")





# 3
def mer(a, b):
    c = a + b
    c = sorted(c)
    return c


a = [1, 2, 3]
b = [2, 4, 5]
print(mer(a,b))




# 4
def cnt(a):
    odd = 0
    even = 0
    for i in a:
        if i % 2 == 0:
            even += 1
        else:
            odd += 1
    return even, odd
            
    
a = [1, 2, 37, 42, 34]
even, odd = cnt(a)
print(f"even:{even}, odd:{odd}")




# 5
def is_valid(pas):
    dow = 0
    upr = 0
    num = 0
    chr = 0
    l = 0
    if len(pas) <= 16 and len(pas) >= 6:
        l = 1
    for c in pas:
        if 'A' <= c and c <= 'Z':
            upr += 1
        elif 'a' <= c and c <= 'z':
            dow += 1
        elif '0' <= c and c <= '9':
            num += 1
        elif c == '#' or c == '$' or c == '@':
            chr += 1
    if upr and dow and num and chr and l:
        return True
    else: 
        return False
    
    
p = '12A#a'
print(is_valid(p))





# 6
def check(c):
    consonant = {'a', 'e', 'i', 'o', 'u'}
    if c in consonant:
        return 'consonant'
    else:
        return 'vowel'
    
c = 'r'
print(check(c))





# 7
def conv(mon):
    dt1 = {"January": 31, "February": 28, "March": 31, "April": 30, "May": 31, "June": 30, "July": 31, "August": 31, "September": 30, "October": 31, "November": 30, "December": 31}
    return dt1[mon]


mon = "June"
print(conv(mon))




# 8
def check(s):
    try: 
        if int(s):
            return True
    except ValueError:
        return False
    
s = '123w'
print(check(s))



# 9 
def check(m, d):
    dt1 = {1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer", 7: "Summer",  8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn", 12: "Winter"}
    return dt1[m]


m = 5
d = 1
print(check(m, d))



# 10
def check(year):
    dt1 = {1: "Monkey", 2: "Ox", 3: "Tiger", 4: "Rabbit", 5: "Dragon", 6: "Snake", 7: "Horse", 8: "Sheep", 9: "Monkey", 10: "Rooster", 11: "Dog", 12: "Pig"}
    y = (year - 5) % 12 + 1
    return dt1[y]


year = 2023
print(check(year))




# 11
def next_day(year, month, day):
    dt1 = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    dt2 = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        dt = dt2
    else:
        dt = dt1
        
    if dt[month] > day:
        return year, month, day+1
    elif dt[month] < day:
        return False
    elif month < 12:
        return year, month+1, 1
    elif month > 12:
        return False
    else:
        return year+1, 1, 1
        


year = 2015
month = 11
day = 30
print(next_day(year, month, day))






# 12
import re

def change(pas):
    pas1 = re.split("\W+", pas)
    pas2 = [x.lower() for x in pas1 if x != '--' and x != '.' and x != ',']
    return pas2


def cnt(pas):
    dtw = {}
    dtl = {}
    for x in pas:
        if x in dtw:
            dtw[x] += 1
        else:
            dtw[x] = 1
        for l in x:
            if l in dtl:
                dtl[l] += 1
            else:
                dtl[l] = 1
    return dtw, dtl


pas = "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal.\
Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this.\
But, in a larger sense, we can not dedicate -- we can not consecrate -- we can not hallow -- this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced. It is rather for us to be here dedicated to the great task remaining before us -- that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion -- that we here highly resolve that these dead shall not have died in vain -- that this nation, under God, shall have a new birth of freedom -- and that government of the people, by the people, for the people, shall not perish from the earth.\
Abraham Lincoln\
November 19, 1863\
"

print(" ".join(change(pas)))
print(cnt(change(pas)))