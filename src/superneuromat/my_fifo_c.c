#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <mpi.h>
#include <errno.h>



char my_fifo[100] = "./my_fifo";

if (mkfifo(my_fifo, 0666) == -1)
{
	if (errno != EEXIST)
	{
		perror("Error in creating fifo_python\n");
		return 1;
	}
}

int fp = open(fifo_python_path, O_RDONLY);

if (fp == -1)
{
	perror("Error opening fifo\n");
	return 2;
}

int buffer;

if (read(fp, &buffer, O_RDONLY) != sizeof(int))
{
	perror("Error reading");
	return 3;
}

print("Received %d", buffer);


close(fp);
