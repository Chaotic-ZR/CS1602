def get_input(fname):
	# 初始化
	f = open(fname, "r")
	clause_num = 0
	num_in_one_clause = 0
	clauses = []

	# 第一行数据读入
	lines = f.readlines()
	first_line = lines[0]
	lst1 = first_line.split()
	clause_num, num_in_one_clause= int(lst1[0]), int(lst1[1])
	clauses = [0 for i in range(clause_num)]

	# 读入clause
	for i in range(1, clause_num + 1):
		line_str = lines[i].split()
		line_int = [int(x) for x in line_str]
		clauses[i-1] = line_int 
	f.close()
	return clauses

a = get_input("1.txt")
print(a)