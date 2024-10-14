#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// Function to compute f(x)
float f(float x) {
    return x * x; // Example: f(x) = x^2
}

// Serial computation of the trapezoidal rule
float trapezoidal_serial(float a, float b, int n) {
    float area = 0.0f;
    float delta_x = (b - a) / n;

    for (int i = 0; i < n; i++) {
        float x1 = a + i * delta_x;
        area += f(x1) + f(x1 + delta_x);
    }

    return area * delta_x / 2.0f;
}

// Parallel computation of the trapezoidal rule
float trapezoidal_parallel(float a, float b, int n, int process_id, int total_processes) {
    float delta_x = (b - a) / n; // Step size for each trapezoid
    float sub_range = (b - a) / total_processes; // Range handled by each process

    // Define local range for the current process
    float local_start = a + process_id * sub_range;
    float local_end = local_start + sub_range;
    float local_sum = 0.0f;

    for (int i = 0; i < n / total_processes; i++) {
        float x1 = local_start + i * delta_x;
        local_sum += f(x1) + f(x1 + delta_x);
    }

    return local_sum * delta_x / 2.0f;
}

int main(int argc, char** argv) {
    int process_id, total_processes;
    float a = 0.0f, b = 1.0f; // Limits of integration
    int n; // Number of trapezoids
    float global_result; // Combined result from all processes
    double serial_time_taken;

    MPI_Init(&argc, &argv); // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id); // Get current process rank
    MPI_Comm_size(MPI_COMM_WORLD, &total_processes); // Get total number of processes

    if (process_id == 0) {
        // Input number of intervals
        printf("Enter the number of intervals: ");
        scanf("%d", &n);

        // Serial computation timing
        clock_t start_time = clock();
        float serial_result = trapezoidal_serial(a, b, n);
        clock_t end_time = clock();
        serial_time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;

        printf("Result (serial): %f\n", serial_result);
        printf("Serial execution time: %f seconds\n", serial_time_taken);
    }

    // Broadcast the number of intervals to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Measure parallel computation time
    double parallel_start = MPI_Wtime();

    // Each process computes its portion of the integral
    float local_result = trapezoidal_parallel(a, b, n, process_id, total_processes);

    // Reduce local results to the global result at the root process
    MPI_Reduce(&local_result, &global_result, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    // End of parallel computation timing
    double parallel_end = MPI_Wtime();

    if (process_id == 0) {
        double parallel_time_taken = parallel_end - parallel_start;
        printf("Result (parallel): %f\n", global_result);
        printf("Parallel execution time: %f seconds\n", parallel_time_taken);

        // Calculate speedup
        double speedup = serial_time_taken / parallel_time_taken;
        printf("Speedup: %f\n", speedup);
    }

    MPI_Finalize(); // Finalize MPI
    return 0;
}
