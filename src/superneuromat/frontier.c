#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <mpi.h>
#include <errno.h>
// #include <omp.h>


/* Compilation instructions

- Basic compilation (uses single precision): gcc -Wall -fPIC -shared -o frontier.so frontier.c
- Want to use double precision: gcc -Wall -DUSE_DOUBLE -FPIC -shared -o frontier.so frontier.c
- Compiling with openmp (do `conda deactivate` on mac first): gcc-14 -fopenmp -Wall -FPIC -shared -o frontier.so frontier.c

*/


// Macro for selecting single vs double precision numbers
// To select double precision at compile time, do: g++ -DUSE_DOUBLE <code>.cpp
#ifdef USE_DOUBLE
	typedef double precision;
#else
	typedef float precision;
#endif 



// Neuron variables
int num_neurons = 0;
precision *neuron_thresholds = NULL;
precision *neuron_leaks = NULL;
precision *neuron_reset_states = NULL;
int *neuron_refractory_periods_original = NULL;
int *neuron_refractory_periods = NULL;
precision *neuron_internal_states = NULL;
precision *spikes = NULL;


// Synapse variables
int num_synapses = 0;
int *pre_synaptic_neuron_ids = NULL;
int *post_synaptic_neuron_ids = NULL;
precision *synaptic_weights = NULL;
int *enable_stdp = NULL;
precision **weights = NULL;
precision **stdp_enabled_synapses = NULL;


// STDP variables
int stdp_time_steps = 0;
precision *stdp_Apos = NULL;
precision *stdp_Aneg = NULL;
int stdp_positive_update = 0;
int stdp_negative_update = 0;


// Input spikes
precision *input_spikes = NULL;



// int initialize_mpi()
// {
// 	/* Initialize MPI

// 	*/

// 	int flag = 0;
// 	MPI_Initialized(&flag);

// 	if (!flag)
// 	{
// 		MPI_Init(NULL, NULL);

// 		return 0;
// 	}

// 	else
// 	{
// 		return 1;
// 	}

// }



// int finalize_mpi()
// {
// 	/* Finalize MPI

// 	*/

// 	int flag = 0;
// 	MPI_Finalized(&flag);

// 	if (!flag)
// 	{
// 		MPI_Finalize();
// 	}

// 	return 0;
// }



// int stdp_setup_frontier(int f_time_steps, precision *f_Apos, precision *f_Aneg, int f_positive_update, int f_negative_update)
// {
// 	/* Setup STDP parameters

// 	Args:
// 		f_time_steps (int): Number of time steps over which STDP learning occurs (default: 3)
// 		f_Apos (precision *): Array of parameters for excitatory STDP updates (default: [1.0, 0.5, 0.25]); number of elements in the list must be equal to time_steps
// 		f_Aneg (precision *): Array of parameters for inhibitory STDP updates (default: [1.0, 0.5, 0.25]); number of elements in the list must be equal to time_steps
// 		positive_update (int): Boolean parameter indicating whether excitatory STDP update should be enabled
// 		negative_update (int): Boolean parameter indicating whether inhibitory STDP update should be enabled

// 	Returns:
// 		0: Everything works well
// 		1: stdp_Apos could not be allocated
// 		2: stdp_Aneg could not be allocated

// 	*/

// 	// printf("[C stdp_setup_frontier] Entered the stdp_setup_frontier function\n");

// 	// Assign fundamental data types
// 	stdp_time_steps = f_time_steps;
// 	stdp_positive_update = f_positive_update;
// 	stdp_negative_update = f_negative_update;

// 	printf("[C stdp_setup_frontier] Assigned fundamental data types\n");


// 	// Allocate memory for derived data types
// 	stdp_Apos = (precision *) malloc(stdp_time_steps * sizeof(precision));

// 	if (stdp_Apos == NULL)
// 	{
// 		printf("[C stdp_setup_frontier] Memory allocation failed for Apos\n");
// 		exit(1);
// 	}

// 	stdp_Aneg = (precision *) malloc(stdp_time_steps * sizeof(precision));
	
// 	if (stdp_Aneg == NULL)
// 	{
// 		printf("[C stdp_setup_frontier] Memory allocation failed for Aneg\n");
// 		exit(2);
// 	}

// 	printf("[C stdp_setup_frontier] Allocated memory for derived data\n");


// 	// Assign derived data types
// 	#pragma omp parallel for
// 	for (int i = 0; i < stdp_time_steps; i++)
// 	{
// 		stdp_Apos[i] = f_Apos[i];
// 		stdp_Aneg[i] = f_Aneg[i];
// 	}

// 	printf("[C stdp_setup_frontier] Assigned derived data\n");


// 	// Print data
// 	printf("[C stdp_setup_frontier] Received following data:\n");
// 	printf("\tstdp_time_steps: %d\n", stdp_time_steps);

// 	printf("\tstdp_Apos: [ ");
// 	for (int i = 0; i < stdp_time_steps; i++)
// 	{
// 		printf("%.4f ", stdp_Apos[i]);
// 	}
// 	printf("]\n");

// 	printf("\tstdp_Aneg: [ ");
// 	for (int i = 0; i < stdp_time_steps; i++)
// 	{
// 		printf("%.4f ", stdp_Aneg[i]);
// 	}
// 	printf("]\n");

// 	printf("\tstdp_positive_update: %d\n", stdp_positive_update);
// 	printf("\tstdp_negative_update: %d\n", stdp_negative_update);


// 	return 0;
// }




// int setup_frontier(int f_num_neurons, precision *f_neuron_thresholds, precision **f_synaptic_weights)
// int setup_frontier(int f_num_neurons, precision *f_neuron_thresholds)
// {
// 	/* Setup the neuromorphic model for simulation on Frontier backend

// 	*/

// 	int rank, size;
// 	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
// 	MPI_Comm_size(MPI_COMM_WORLD, &size);	

// 	printf("[C setup_frontier] Hello from %d out of %d\n", rank, size);

// 	return 0;

// }




// precision *simulate_frontier()
// {
	// Allocate result
	// precision *result = (precision *) malloc(first_data_size * sizeof(precision));

	// if (result == NULL)
	// {
	// 	printf("Memory allocation failed for result\n");
	// }


	// // Print first_data
	// printf("[C simulate_frontier] First data: [ ");

	// for (int i = 0; i < first_data_size; i++)
	// {
	// 	printf("%.2f ", first_data[i]);
	// }

	// printf("]\n");


	// // Print second_data	
	// printf("[C simulate_frontier] First data: [ ");

	// for (int i = 0; i < second_data_size; i++)
	// {
	// 	printf("%.2f ", second_data[i]);
	// }

	// printf("]\n");


	// // Perform computation	
	// // #pragma omp parallel for
	// for (int i = 0; i < first_data_size; i++)
	// {
	// 	result[i] = first_data[i] * second_data[i];
	// }

	
	// printf("[C simulate_frontier] Returning following data: [ ");

	// for (int i = 0; i < first_data_size; i++)
	// {
	// 	printf("%.2f ", result[i]);
	// }

	// printf("]\n");


// 	return NULL;

// }




int free_memory()
{	// Free up all dynamically allocated memory

	// free(first_data);
	// first_data = NULL;

	// free(second_data);
	// second_data = NULL;

	return 0;
}



int main(int argc, char **argv)
{
	// Initialize MPI
	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	printf("Hello from process %d out of %d\n", rank, size);
	// printf("size of int: %lu\n", sizeof(int));


	// Initialize variable to receive from Python
	// int array_size;
	// precision *stdp_Apos;


	// Receive data from Python
	if (rank == 0)
	{
		// printf("Number of arguments: %d\n", argc);

		// for (int i = 0; i < argc; i++)
		// 	printf("Argument %d: %s\n", i, argv[i]);

		// Find the path of the fifo
		char *fifo_python_path = argv[1];
		char *fifo_c_path = argv[2];

		printf("[C rank %d of %d] fifo_python_path: %s\n", rank, size, fifo_python_path);
		printf("[C rank %d of %d] fifo_c_path: %s\n", rank, size, fifo_c_path);


		// Wait for fifo_python to open 
		// while (1)
		// {
		// 	if (access(fifo_python_path, F_OK) == 0)
		// 	{
		// 		break;
		// 	}

		// 	sleep(5);
		// }


		// Create fifo_python to send data from Python to C
		if (mkfifo(fifo_python_path, 0666) == -1)
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
			perror("Error opening fifo_python\n");
			return 2;
		}

		printf("[C rank %d of %d] fifo_python opened successfully\n", rank, size);


		// Read stdp_time_steps from fifo_python
		if (read(fp, &stdp_time_steps, sizeof(int)) != sizeof(int))
		{
			perror("Error reading stdp_time_steps from fifo_python\n");
			close(fp);
			return 2;
		}

		printf("[C rank %d of %d] stdp_time_steps: %d\n", rank, size, stdp_time_steps);


		// Allocate memory for stdp_Apos
		stdp_Apos = (precision *) malloc(stdp_time_steps * sizeof(precision));

		if (stdp_Apos == NULL)
		{
			perror("Error allocating memory for stdp_Apos");
			close(fp);
			return 3;
		}

		// Read stdp_Apos from fifo_python
		if (read(fp, stdp_Apos, stdp_time_steps * sizeof(precision)) != stdp_time_steps * sizeof(precision))
		{
			perror("Error reading stdp_Apos from fifo_python\n");
			close(fp);
			return 4;
		}

		printf("[C rank %d of %d] stdp_Apos: [ ", rank, size);
		for (int i = 0; i < stdp_time_steps; i++)
		{
			printf("%f ", stdp_Apos[i]);
		}
		printf("]\n");


		// Allocate memory for stdp_Aneg
		stdp_Aneg = (precision *) malloc(stdp_time_steps * sizeof(precision));

		if (stdp_Aneg == NULL)
		{
			perror("Error allocating memory for stdp_Aneg");
			close(fp);
			return 5;
		}

		// Read stdp_Aneg from fifo_python
		if (read(fp, stdp_Aneg, stdp_time_steps * sizeof(precision)) != stdp_time_steps * sizeof(precision))
		{
			perror("Error reading stdp_Aneg from fifo_python\n");
			close(fp);
			return 6;
		}

		printf("[C rank %d of %d] stdp_Aneg: [ ", rank, size);
		for (int i = 0; i < stdp_time_steps; i++)
		{
			printf("%f ", stdp_Aneg[i]);
		}
		printf("]\n");


		// Read stdp_positive_update from fifo_python
		if (read(fp, &stdp_positive_update, sizeof(int)) != sizeof(int))
		{
			perror("Error reading stdp_positive_update from fifo_python\n");
			close(fp);
			return 7;
		}

		printf("[C rank %d of %d] stdp_positive_update: %d\n", rank, size, stdp_positive_update);


		// Read stdp_negative_update from fifo_python
		if (read(fp, &stdp_negative_update, sizeof(int)) != sizeof(int))
		{
			perror("Error reading stdp_negative_update from fifo_python\n");
			close(fp);
			return 8;
		}

		printf("[C rank %d of %d] stdp_negative_update: %d\n", rank, size, stdp_negative_update);


		fp = open(fifo_python_path, O_RDONLY);

		if (fp == -1)
		{
			perror("Error opening fifo_python\n");
			return 9;
		}

		printf("[C rank %d of %d] fifo_python opened successfully\n", rank, size);

		
		// Read num_neurons
		if(read(fp, &num_neurons, sizeof(int)) != sizeof(int))
		{
			perror("Error reading num_neurons from fifo_python");
			close(fp);
			return 10;
		}

		printf("[C rank %d of %d] num_neurons: %d\n", rank, size, num_neurons);


		// Allocate memory for neuron_thresholds
		neuron_thresholds = (precision *) malloc(num_neurons * sizeof(precision));

		if (neuron_thresholds == NULL)
		{
			perror("Error allocating memory for neuron_thresholds");
			close(fp);
			return 11;
		}

		// Read neuron_thresholds from fifo_python
		if (read(fp, neuron_thresholds, num_neurons * sizeof(precision)) != num_neurons * sizeof(precision))
		{
			perror("Error reading neuron_thresholds from fifo_python\n");
			close(fp);
			return 12;
		}

		printf("[C rank %d of %d] neuron_thresholds: [ ", rank, size);
		for (int i = 0; i < num_neurons; i++)
		{
			printf("%f ", neuron_thresholds[i]);
		}
		printf("]\n");


		// Allocate memory for neuron_leaks
		neuron_leaks = (precision *) malloc(num_neurons * sizeof(precision));

		if (neuron_leaks == NULL)
		{
			perror("Error allocating memory for neuron_leaks");
			close(fp);
			return 13;
		}

		// Read neuron_leaks from fifo_python
		if (read(fp, neuron_leaks, num_neurons * sizeof(precision)) != num_neurons * sizeof(precision))
		{
			perror("Error reading neuron_leaks from fifo_python\n");
			close(fp);
			return 14;
		}

		printf("[C rank %d of %d] neuron_leaks: [ ", rank, size);
		for (int i = 0; i < num_neurons; i++)
		{
			printf("%f ", neuron_leaks[i]);
		}
		printf("]\n");


		// Allocate memory for neuron_reset_states
		neuron_reset_states = (precision *) malloc(num_neurons * sizeof(precision));

		if (neuron_reset_states == NULL)
		{
			perror("Error allocating memory for neuron_reset_states");
			close(fp);
			return 15;
		}

		// Read neuron_reset_states from fifo_python
		if (read(fp, neuron_reset_states, num_neurons * sizeof(precision)) != num_neurons * sizeof(precision))
		{
			perror("Error reading neuron_reset_states from fifo_python\n");
			close(fp);
			return 16;
		}

		printf("[C rank %d of %d] neuron_reset_states: [ ", rank, size);
		for (int i = 0; i < num_neurons; i++)
		{
			printf("%f ", neuron_reset_states[i]);
		}
		printf("]\n");


		// Allocate memory for neuron_refractory_periods
		neuron_refractory_periods = (int *) malloc(num_neurons * sizeof(int));

		if (neuron_refractory_periods == NULL)
		{
			perror("Error allocating memory for neuron_refractory_periods");
			close(fp);
			return 17;
		}

		// Read neuron_refractory_periods from fifo_python
		if (read(fp, neuron_refractory_periods, num_neurons * sizeof(int)) != num_neurons * sizeof(int))
		{
			perror("Error reading neuron_refractory_periods from fifo_python\n");
			close(fp);
			return 18;
		}

		printf("[C rank %d of %d] neuron_refractory_periods: [ ", rank, size);
		for (int i = 0; i < num_neurons; i++)
		{
			printf("%d ", neuron_refractory_periods[i]);
		}
		printf("]\n");


		
		// Read num_synapses
		if(read(fp, &num_synapses, sizeof(int)) != sizeof(int))
		{
			perror("Error reading num_synapses from fifo_python");
			close(fp);
			return 19;
		}

		printf("[C rank %d of %d] num_synapses: %d\n", rank, size, num_synapses);


		// Allocate memory for pre_synaptic_neuron_ids 
		pre_synaptic_neuron_ids = (int *) malloc(num_synapses * sizeof(int));

		if (pre_synaptic_neuron_ids == NULL)
		{
			perror("Error allocating memory for pre_synaptic_neuron_ids");
			close(fp);
			return 17;
		}

		// Read pre_synaptic_neuron_ids from fifo_python
		if (read(fp, pre_synaptic_neuron_ids, num_synapses * sizeof(int)) != num_synapses * sizeof(int))
		{
			perror("Error reading pre_synaptic_neuron_ids from fifo_python\n");
			close(fp);
			return 18;
		}

		printf("[C rank %d of %d] pre_synaptic_neuron_ids: [ ", rank, size);
		for (int i = 0; i < num_synapses; i++)
		{
			printf("%d ", pre_synaptic_neuron_ids[i]);
		}
		printf("]\n");


		// Allocate memory for post_synaptic_neuron_ids 
		post_synaptic_neuron_ids = (int *) malloc(num_synapses * sizeof(int));

		if (post_synaptic_neuron_ids == NULL)
		{
			perror("Error allocating memory for post_synaptic_neuron_ids");
			close(fp);
			return 17;
		}

		// Read post_synaptic_neuron_ids from fifo_python
		if (read(fp, post_synaptic_neuron_ids, num_synapses * sizeof(int)) != num_synapses * sizeof(int))
		{
			perror("Error reading post_synaptic_neuron_ids from fifo_python\n");
			close(fp);
			return 18;
		}

		printf("[C rank %d of %d] post_synaptic_neuron_ids: [ ", rank, size);
		for (int i = 0; i < num_synapses; i++)
		{
			printf("%d ", post_synaptic_neuron_ids[i]);
		}
		printf("]\n");


		// Allocate memory for synaptic weights 
		synaptic_weights = (precision *) malloc(num_synapses * sizeof(precision));

		if (synaptic_weights == NULL)
		{
			perror("Error allocating memory for synaptic_weights");
			close(fp);
			return 17;
		}

		// Read synaptic weights from fifo_python
		if (read(fp, synaptic_weights, num_synapses * sizeof(precision)) != num_synapses * sizeof(precision))
		{
			perror("Error reading synaptic_weights from fifo_python\n");
			close(fp);
			return 18;
		}

		printf("[C rank %d of %d] synaptic_weights: [ ", rank, size);
		for (int i = 0; i < num_synapses; i++)
		{
			printf("%f ", synaptic_weights[i]);
		}
		printf("]\n");


		// Allocate memory for enable_stdp 
		enable_stdp = (int *) malloc(num_synapses * sizeof(int));

		if (enable_stdp == NULL)
		{
			perror("Error allocating memory for enable_stdp");
			close(fp);
			return 17;
		}

		// Read enable_stdp from fifo_python
		if (read(fp, enable_stdp, num_synapses * sizeof(int)) != num_synapses * sizeof(int))
		{
			perror("Error reading enable_stdp from fifo_python\n");
			close(fp);
			return 18;
		}

		printf("[C rank %d of %d] enable_stdp: [ ", rank, size);
		for (int i = 0; i < num_synapses; i++)
		{
			printf("%d ", enable_stdp[i]);
		}
		printf("]\n");









		// Create fifo_c to send data from C to Python
		// if (mkfifo(fifo_c_path, 0666) == -1)
		// {
		// 	if (errno != EEXIST)
		// 	{
		// 		perror("Error creating fifo_c\n");
		// 		return 3;
		// 	}
		// }

		// int fc = open(fifo_c_path, O_WRONLY);

		// if (fc == -1)
		// {
		// 	perror("Error opening fifo_c\n");
		// 	return 4;
		// }

		// printf("[C rank %d of %d] fifo_c opened successfully\n", rank, size);

		
		// // Send a message to Python
		// char* flag = "ABCDE";
		// if (write(fc, &flag, sizeof(flag)) == -1)
		// {
		// 	perror("Error writing to fifo_c");
		// 	return 1;
		// }


		// Close the fifos
		// close(fc);
		close(fp);




		// Free dynamic memory
		free(neuron_thresholds);
		free(neuron_leaks);
		free(neuron_reset_states);
		// free(neuron_refractory_periods_original);
		free(neuron_refractory_periods);
		// free(neuron_internal_states);
		// free(spikes);
		free(pre_synaptic_neuron_ids);
		free(post_synaptic_neuron_ids);
		free(synaptic_weights);
		free(enable_stdp);
		// free(weights);
		// free(stdp_enabled_synapses);
		free(stdp_Apos);
		free(stdp_Aneg);
		// free(input_spikes);

	}


	MPI_Finalize();

	return 0;
}




















