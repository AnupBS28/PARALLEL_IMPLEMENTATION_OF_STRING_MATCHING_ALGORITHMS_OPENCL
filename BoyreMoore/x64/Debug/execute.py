import subprocess
for i in range (14):
    print ("Test ",i)
    print("\n")
    # #subprocess.call(["gcc",cmd]) #For Compiling
    f = open('output.txt', 'a')
    f.write("Test "+str(i))
    f.write("\n")
    f.close()
    call = "BoyreMoore"
    # print(call)
    subprocess.call("BoyreMoore")