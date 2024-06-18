
/* 3D Graphics using ASCII graphics with OpenCL integration for matrix multiplication */
// #define NOGRAPHICS 1

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#ifndef NOGRAPHICS
#include <unistd.h>
#include <ncurses.h>
#endif

#define DELAY 10000
	// maximum screen size, both height and width
#define SCREENSIZE 100
	// default number of iterations to run before exiting, only used
	// when graphics are turned off
#define ITERATIONS 1000

	// number of points
int pointCount;
	// array of points before transformation
float *pointArray;
	// array of points after transformation
float *drawArray;

float transformMatrix[16]; // This array will be used to store the transposed matrix
	// transformation matrix
float transformArray[4][4];
	// display buffers
char frameBuffer[SCREENSIZE][SCREENSIZE];
float depthBuffer[SCREENSIZE][SCREENSIZE];

// Global variables for OpenCL components
cl_context context;
cl_command_queue commandQueue;
cl_command_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
cl_program program;
cl_kernel kernel;
cl_mem pointArrayBuffer, transformArrayBuffer, drawArrayBuffer;
cl_int err;

void vectorMult(float a[4], float b[4], float c[4][4]);

#ifndef NOGRAPHICS
	// maximum screen dimensions
int max_y = 0, max_x = 0;
#endif

#ifndef NOGRAPHICS
int drawPoints() {
int c, i, j;
float multx, multy;
float point[4];

	// update screen maximum size
   getmaxyx(stdscr, max_y, max_x);

	// used to scale position of points based on screen size
   multx = (float)max_x / SCREENSIZE;
   multy = (float)max_y / SCREENSIZE;

   clear();

	// display points
   for(i=0; i<SCREENSIZE; i++) {
      for(j=0; j<SCREENSIZE; j++) {
         if (frameBuffer[i][j] == 'X')
            mvprintw(i * multy, j*multx, "X");
         else if (frameBuffer[i][j] == 'o')
            mvprintw(i * multy, j*multx, "o");
         else if (frameBuffer[i][j] == '.')
            mvprintw(i * multy, j*multx, ".");
      }
   }

   refresh();

   usleep(DELAY);

	// read keyboard and exit if 'q' pressed
   c = getch();
   if (c == 'q') return(1);
   else return(0);
}
#endif

	/* calculates the product of matrices b and c
           stores the result in matrix a */
void matrixMult(float a[4][4], float b[4][4], float c[4][4]) {
int row, col, element;

   for (row=0; row<4; row++) {
      for (col=0; col<4; col++) {
         a[row][col] = 0.0; 
         for (element=0; element<4; element++) {
            a[row][col] += b[row][element] * c[element][col]; 
         }
      }
   }
}


	/* calculates the product of vector b and matrix c
           stores the result in vector a */
void vectorMult(float a[4], float b[4], float c[4][4]) {
int col, element;

   for (col=0; col<4; col++) {
      a[col] = 0.0; 
      for (element=0; element<4; element++) {
         a[col] += b[element] * c[element][col]; 
      }
   }
}

void allocateArrays() {
// Allocate contiguous memory for pointArray
pointArray = malloc(sizeof(float) * pointCount * 4); // 4 components per point

// Allocate contiguous memory for drawArray
drawArray = malloc(sizeof(float) * pointCount * 4); // 4 components per point

}

void cubePointArray() {
    // Point 0
    pointArray[0 * 4 + 0] = 0.5;
    pointArray[0 * 4 + 1] = 0.0;
    pointArray[0 * 4 + 2] = 0.5;
    pointArray[0 * 4 + 3] = 1.0;

    // Point 1
    pointArray[1 * 4 + 0] = 0.5;
    pointArray[1 * 4 + 1] = 0.0;
    pointArray[1 * 4 + 2] = -0.5;
    pointArray[1 * 4 + 3] = 1.0;

    // Point 2
    pointArray[2 * 4 + 0] = -0.5;
    pointArray[2 * 4 + 1] = 0.0;
    pointArray[2 * 4 + 2] = -0.5;
    pointArray[2 * 4 + 3] = 1.0;

    // Point 3
    pointArray[3 * 4 + 0] = -0.5;
    pointArray[3 * 4 + 1] = 0.0;
    pointArray[3 * 4 + 2] = 0.5;
    pointArray[3 * 4 + 3] = 1.0;

    // Point 4
    pointArray[4 * 4 + 0] = 0.5;
    pointArray[4 * 4 + 1] = 1.0;
    pointArray[4 * 4 + 2] = 0.5;
    pointArray[4 * 4 + 3] = 1.0;

    // Point 5
    pointArray[5 * 4 + 0] = 0.5;
    pointArray[5 * 4 + 1] = 1.0;
    pointArray[5 * 4 + 2] = -0.5;
    pointArray[5 * 4 + 3] = 1.0;

    // Point 6
    pointArray[6 * 4 + 0] = -0.5;
    pointArray[6 * 4 + 1] = 1.0;
    pointArray[6 * 4 + 2] = -0.5;
    pointArray[6 * 4 + 3] = 1.0;

    // Point 7
    pointArray[7 * 4 + 0] = -0.5;
    pointArray[7 * 4 + 1] = 1.0;
    pointArray[7 * 4 + 2] = 0.5;
    pointArray[7 * 4 + 3] = 1.0;
}


void randomPointArray() {
    int i;
    float val;

    for (i = 0; i < pointCount; i++) {
        int baseIndex = i * 4; // Calculate the base index for point i

        val = (float) random() / 10000.0;
        pointArray[baseIndex] = 2.5 * ((val - trunc(val)) - 0.5);

        val = (float) random() / 10000.0;
        pointArray[baseIndex + 1] = 2.5 * ((val - trunc(val)) - 0.5);

        val = (float) random() / 10000.0;
        pointArray[baseIndex + 2] = 2.5 * ((val - trunc(val)) - 0.5);

        val = (float) random() / 10000.0;
        pointArray[baseIndex + 3] = 2.5 * ((val - trunc(val)) - 0.5); 
    }
}


void initTransform() {
int i, j; 

   for (i=0; i<4; i++)
      for (j=0; j<4; j++)
        if (i == j)
           transformArray[i][j] = 1.0;
        else
           transformArray[i][j] = 0.0;
}


void xRot(int rot) {
float oneDegree = 0.017453;
float angle, sinAngle, cosAngle;
float result[4][4];
int i, j;
float rotx[4][4]  = {1.0, 0.0, 0.0, 0.0, 
                     0.0, 1.0, 0.0, 0.0, 
                     0.0, 0.0, 1.0, 0.0, 
                     0.0, 0.0, 0.0, 1.0}; 

   angle = (float) rot * oneDegree;
   sinAngle = sinf(angle);
   cosAngle = cosf(angle);

   rotx[1][1] = cosAngle;
   rotx[2][2] = cosAngle;
   rotx[1][2] = -sinAngle;
   rotx[2][1] = sinAngle;

   matrixMult(result, transformArray, rotx);

	// copy result to transformation matrix
   for (i=0; i<4; i++)
      for (j=0; j<4; j++)
         transformArray[i][j] = result[i][j];
}

void yRot(int rot) {
float oneDegree = 0.017453;
float angle, sinAngle, cosAngle;
float result[4][4];
int i, j;
float roty[4][4]  = {1.0, 0.0, 0.0, 0.0, 
                     0.0, 1.0, 0.0, 0.0, 
                     0.0, 0.0, 1.0, 0.0, 
                     0.0, 0.0, 0.0, 1.0}; 

   angle = (float) rot * oneDegree;
   sinAngle = sinf(angle);
   cosAngle = cosf(angle);

   roty[0][0] = cosAngle;
   roty[2][2] = cosAngle;
   roty[0][2] = sinAngle;
   roty[2][0] = -sinAngle;

   matrixMult(result, transformArray, roty);

	// copy result to transformation matrix
   for (i=0; i<4; i++)
      for (j=0; j<4; j++)
         transformArray[i][j] = result[i][j];
}

void zRot(int rot) {
float oneDegree = 0.017453;
float angle, sinAngle, cosAngle;
float result[4][4];
int i, j;
float rotz[4][4]  = {1.0, 0.0, 0.0, 0.0, 
                     0.0, 1.0, 0.0, 0.0, 
                     0.0, 0.0, 1.0, 0.0, 
                     0.0, 0.0, 0.0, 1.0}; 

   angle = (float) rot * oneDegree;
   sinAngle = sinf(angle);
   cosAngle = cosf(angle);

   rotz[0][0] = cosAngle;
   rotz[1][1] = cosAngle;
   rotz[0][1] = -sinAngle;
   rotz[1][0] = sinAngle;

   matrixMult(result, transformArray, rotz);

	// copy result to transformation matrix
   for (i=0; i<4; i++)
      for (j=0; j<4; j++)
         transformArray[i][j] = result[i][j];
}

void translate(float x, float y, float z) {
   transformArray[3][0] = x;
   transformArray[3][1] = y;
   transformArray[3][2] = z;
}

void clearBuffers() {
int i, j;

	// empty the frame buffer
	// set the depth buffer to a large distance
   for(i=0; i<SCREENSIZE; i++) {
      for(j=0; j<SCREENSIZE; j++) {
         frameBuffer[i][j] = ' ';
         depthBuffer[i][j] = -1000.0; 
      }
   }
}

// OpenCL kernel for matrix multiplication
const char *matrixMultiplicationKernelSource = 
    "__kernel void matrix_multiply(__global const float4 *points, __global const float *transMatrix, __global float4 *transformedPoints, const int pointCount) {\n"
    "    int i = get_global_id(0);\n"
    "    if (i < pointCount) {\n"
    "        float4 point = points[i];\n"
    "        transformedPoints[i].x = dot(point, vload4(0, transMatrix));\n"
    "        transformedPoints[i].y = dot(point, vload4(4, transMatrix));\n"
    "        transformedPoints[i].z = dot(point, vload4(8, transMatrix));\n"
    "        transformedPoints[i].w = dot(point, vload4(12, transMatrix));\n"
    "    }\n"
    "}\n";




void handleOpenCLError(cl_int err, const char *message) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "%s: OpenCL error code %d\n", message, err);
        exit(EXIT_FAILURE);
    }
}


// Modified movePoints function using OpenCL
void movePoints() {
    static int counter = 1;
    int i, j, x, y;
    size_t globalWorkSize[1] = {pointCount};
   
    // Initialize transformation matrix
    initTransform();

    // Add some rotations to the transformation matrix
    xRot(counter);
    yRot(counter);
    counter++;


    for (i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            transformMatrix[j * 4 + i] = transformArray[i][j];
        }
    }

    // Write the transposed matrix to the device buffer
   err = clEnqueueWriteBuffer(commandQueue, transformArrayBuffer, CL_TRUE, 0, sizeof(float) * 16, transformMatrix, 0, NULL, NULL);
   handleOpenCLError(err, "Failed to write to transformArrayBuffer");

   err = clEnqueueWriteBuffer(commandQueue, pointArrayBuffer, CL_TRUE, 0, sizeof(float) * pointCount * 4, pointArray, 0, NULL, NULL);
   handleOpenCLError(err, "Failed to write to pointArrayBuffer");

   // Set kernel arguments
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &pointArrayBuffer);
   err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &transformArrayBuffer);
   err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &drawArrayBuffer);
   err |= clSetKernelArg(kernel, 3, sizeof(int), &pointCount);
   handleOpenCLError(err, "Failed to set kernel arguments");

   // Execute the kernel
   err = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
   handleOpenCLError(err, "Failed to execute kernel");

   // Read back the results into drawArray
   err = clEnqueueReadBuffer(commandQueue, drawArrayBuffer, CL_TRUE, 0, sizeof(float) * pointCount * 4, drawArray, 0, NULL, NULL);
   handleOpenCLError(err, "Failed to read from drawArrayBuffer");
   // Execute the kernel
   err = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
   if (err != CL_SUCCESS) {
      printf("Error executing kernel: %d\n", err);
      return;
   }

   // Read back the results into drawArray
   err = clEnqueueReadBuffer(commandQueue, drawArrayBuffer, CL_TRUE, 0, sizeof(float) * pointCount * 4, drawArray, 0, NULL, NULL);
   if (err != CL_SUCCESS) {
      printf("Error reading drawArrayBuffer: %d\n", err);
      return;
   }

    // Scale the points for curses screen resolution
   for (i = 0; i < pointCount; i++) {
      int baseIndex = i * 4; // Calculate the base index for point i

      // Validation
      if (drawArray[baseIndex] < -1000 || drawArray[baseIndex] > 1000 || 
         drawArray[baseIndex + 1] < -1000 || drawArray[baseIndex + 1] > 1000 || 
         drawArray[baseIndex + 2] < -1000 || drawArray[baseIndex + 2] > 1000) {
         printf("Invalid coordinates for point %d: %f, %f, %f\n", 
                  i, drawArray[baseIndex], drawArray[baseIndex + 1], drawArray[baseIndex + 2]);
      }

      // Scaling
      drawArray[baseIndex] = (drawArray[baseIndex] + 1) * (SCREENSIZE / 2);
      drawArray[baseIndex + 1] = (drawArray[baseIndex + 1] + 1) * (SCREENSIZE / 2);

   }

    // Clear buffers before drawing screen
    clearBuffers();

    // Draw the screen
   for (i = 0; i < pointCount; i++) {
      int baseIndex = i * 4; // Calculate the base index for point i

      x = (int) drawArray[baseIndex];     // Accessing x coordinate
      y = (int) drawArray[baseIndex + 1]; // Accessing y coordinate
      float z = drawArray[baseIndex + 2]; // Accessing z coordinate

      if (x >= 0 && x < SCREENSIZE && y >= 0 && y < SCREENSIZE) {
         if (depthBuffer[x][y] < z) { 
               if (z > 60.0)
                  frameBuffer[x][y] = 'X'; 
               else if (z < 40.0)
                  frameBuffer[x][y] = '.'; 
               else
                  frameBuffer[x][y] = 'o'; 
               depthBuffer[x][y] = z;
         }
      }
   }


}



int main(int argc, char *argv[]) {
int i, j, count;
int argPtr;
int drawCube, drawRandom;

// set number of iterations, only used for timing tests 
// not used in curses version
count = ITERATIONS;

// initialize drawing objects
drawCube = 0;
drawRandom = 0;

// read command line arguments for number of iterations 
if (argc > 1) {
   argPtr = 1;
   while(argPtr < argc) {
      if (strcmp(argv[argPtr], "-i") == 0) {
         sscanf(argv[argPtr+1], "%d", &count);
         argPtr += 2;
      } else if (strcmp(argv[argPtr], "-cube") == 0) {
         drawCube = 1;
         pointCount = 8;
         argPtr += 1;
      } else if (strcmp(argv[argPtr], "-points") == 0) {
         drawRandom = 1;
         sscanf(argv[argPtr+1], "%d", &pointCount);
         argPtr += 2;
      } else {
         printf("USAGE: %s <-i iterations> <-cube | -points #>\n", argv[0]);
         printf(" iterations -the number of times the population will be updated\n");
      printf("    the number of iterations only affects the non-curses program\n");
      printf(" the curses program exits when q is pressed\n");
      printf(" choose either -cube to draw the cube shape or -points # to\n");
      printf("    draw random points where # is an integer number of points to draw\n");
         exit(1);
      }
   }
}

// allocate space for arrays to store point position
allocateArrays();
if (drawCube == 1)
   cubePointArray();
else if (drawRandom == 1)
   randomPointArray();
else {
   printf("You must choose either <-cube> or <-points #> on the command line.\n");
   exit(1);
}

// Set up OpenCL

const char *kernelSource = matrixMultiplicationKernelSource;
cl_platform_id platform_id = NULL;
cl_device_id device_id = NULL;
cl_uint ret_num_devices;
cl_uint ret_num_platforms;
err = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
err |= clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);

commandQueue = clCreateCommandQueueWithProperties(context, device_id, properties, &err);
handleOpenCLError(err, "Failed to create command queue");

program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
 handleOpenCLError(err, "Failed to create program");

err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
handleOpenCLError(err, "Failed to build program");

kernel = clCreateKernel(program, "matrix_multiply", &err);
handleOpenCLError(err, "Failed to create kernel");

// Create memory buffers
pointArrayBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * pointCount * 4, NULL, &err);
handleOpenCLError(err, "Failed to create pointArrayBuffer");

transformArrayBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 16, NULL, &err);
handleOpenCLError(err, "Failed to create transformArrayBuffer");

drawArrayBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * pointCount * 4, NULL, &err);
handleOpenCLError(err, "Failed to create drawArrayBuffer");

// Transpose the matrix from transformArray (row-major) to transMatrix (column-major)
for (i = 0; i < 4; i++) {
    for (j = 0; j < 4; j++) {
        transformMatrix[j * 4 + i] = transformArray[i][j];
    }
}



#ifndef NOGRAPHICS
	// initialize ncurses
   initscr();
   noecho();
   cbreak();
   timeout(0);
   curs_set(FALSE);
     // Global var `stdscr` is created by the call to `initscr()`
   getmaxyx(stdscr, max_y, max_x);
#endif


	// draw and move points using ncurses 
	// do not calculate timing in this loop, ncurses will reduce performance
#ifndef NOGRAPHICS
   while(1) {
      if (drawPoints() == 1) break;
      movePoints();
   }
#endif

	// calculate movement of points but do not use ncurses to draw
#ifdef NOGRAPHICS
   printf("Number of iterations %d\n", count);


   for(i=0; i<count; i++) {
      movePoints();
   }
#endif 

// Release OpenCL resources
clReleaseMemObject(pointArrayBuffer);
clReleaseMemObject(transformArrayBuffer);
clReleaseMemObject(drawArrayBuffer);
clReleaseKernel(kernel);
clReleaseProgram(program);
clReleaseCommandQueue(commandQueue);
clReleaseContext(context);

#ifndef NOGRAPHICS
	// shut down ncurses
   endwin();
#endif

}

