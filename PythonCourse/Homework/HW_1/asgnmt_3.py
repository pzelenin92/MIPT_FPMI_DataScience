"""Нужно вывести первые n строк треугольника Паскаля. В этом треугольнике на вершине и по бокам стоят единицы,
а каждое число внутри равно сумме двух расположенных над ним чисел. Вид треугольника можно не форматировать, т. е. просто
выводить на печать строку из треугольника Паскаля."""

n = 15

print("line number 0:",1)
print("line number 1:",1,1)

prev_list = [1,1]
for i in range(2,n+1):
    len_prev_list= len(prev_list)
    next_list = []
    next_list.append(1)
    for x in range(0,len(prev_list)):
        if x+1 <len_prev_list:
            next_number = prev_list[x]+prev_list[x+1]
            next_list.append(next_number)
    next_list.append(1)
    prev_list=next_list
    print("line number "+str(i)+" :",end=" ")
    for a in prev_list:
        print(a,end=" ")
    print()