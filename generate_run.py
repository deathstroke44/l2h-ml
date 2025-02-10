cpus=[1,2,3,4,5,6,7,8,9,10]

tmp='taskset --cpu-list [cpu]-[cpu] python3 [script] & '

for i in range(1,11):
    pf='test-'+str(i)+'.py'
    cpu=str(cpus[i-1])
    cmd=tmp.replace('[cpu]',cpu).replace('[script]',pf)
    print(cmd,end='')
print('echo 1')