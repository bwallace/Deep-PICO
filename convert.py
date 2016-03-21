__author__ = 'zhangye'
#convert abstract file into a set of files
#each file is a single abstract
def file_to_sen(file="intervention-parsing/abstracts_2.txt",out_dir="abstracts2/"):
    with open(file,'rb') as files:
        file_no = 0
        write_file = None
        for i, f in enumerate(files):
            if("- abstract:" in f):
               if(file_no!=0):
                   write_file.close()
                   file_no += 1
                   write_file = open(out_dir+str(file_no)+".txt",'wb')
               else:
                   file_no += 1
                   write_file = open(out_dir+str(file_no)+".txt",'wb')
            else:
               write_file.write(f)
        write_file.close()    #close the last file

def main():
    file_to_sen()
if __name__ == '__main__':
    main()
