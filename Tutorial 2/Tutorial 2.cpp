/*
CMP3752M Parallel Programming
19697602
Daniel Lowe

The program performs image histogram contrast equalisation using open cl and C++. 
The program takes the image input data, and creates a histogram based upon the intensity values. 
An additional implementation is that a kernel (“hist_simple_variable_bins”) also works with a variable number of bins. 
Then the scan add kernel is called which creates an accumulative histogram based upon the intensity values of the image. 
The program continues to call the kernels block sum, scan add atomic and scan add adjust to achieve a scan pattern. 
This is then normalised using division so that the numbers range from the values of 0-255. 
Finally, back projection is performed where the intensity of the image pixels is set to the values in the lookup table. 
This is assigned to an output image which is subsequently displayed to the screen. The resulting image has its contrast adjusted accordingly. 
The execution time of each kernel is outputted and accumulated providing a total execution time. 
This is optimised by being performed in a separate function that takes the event of each kernel and returns the accumulated time so that it can be outputted.
*/

//Imports the necessary libraries
#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

float executionTime(float totalExecutionTime, cl::Event prof_event) {
	//Appends the execution time
	//Gets the execution time of the kernel whilst also appending the total execution time to a variable
	totalExecutionTime += prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
		prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

	
	std::cout << "Kernel execution time [ns]:" <<
		prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
		prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;


	std::cout << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US)
		<< std::endl;

	return totalExecutionTime; //Returns the totalExecutionTime so that it can be accumulated and displayed
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	//Assigns variable for the file name
	string image_filename = "test.pgm";

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}


	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input,"input");


		//Part 3 - host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

	
		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//create an event to monitor execution time
		cl::Event prof_event;

		//3.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try { 
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Part 4 - device operations
		typedef int type;
		//Vector variables for each step
		std::vector<type>histogramVector(256); 
		std::vector<type>blockSumVector(256);
		std::vector<type>scanAddVector(256);
		std::vector<type>scanAddAtomicVector(256);
		std::vector<type>scanAddAdjustVector(256);
		std::vector<type>normalisedVector(256);

		//Variable to keep track of total execution time
		float totalExecutionTime = 0.f;


		size_t outputSize = histogramVector.size() * sizeof(type);
		size_t localSize = 32;
		size_t paddingSize = histogramVector.size() % localSize;
		


			if (paddingSize) {
				std::vector<int> A(localSize - paddingSize, 0);



			}

		//device - buffers

		//Buffers for image input and output
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size()); //should be the same as input image

		//Buffers created for each step of the image histogram constrast equalisation
		cl::Buffer histogramBuffer(context, CL_MEM_READ_WRITE, outputSize);
		cl::Buffer scanAddBuffer(context, CL_MEM_READ_WRITE, outputSize);
		cl::Buffer scanAddAtomicBuffer(context, CL_MEM_READ_WRITE, outputSize);
		cl::Buffer blockSumBuffer(context, CL_MEM_READ_WRITE, outputSize);
		cl::Buffer normaliseAndScale(context, CL_MEM_READ_WRITE, outputSize);


		//4.1 Copy images to device memory
		//write to a buffer object from host memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]); //Queues a write buffer using the image input size and data
		queue.enqueueFillBuffer(histogramBuffer, 0, 0, outputSize); //Fills using the histogram buffer
		cl::Kernel histKernel = cl::Kernel(program, "hist_simple"); //Creates a kernel instance using the hist_simple kernel
		histKernel.setArg(0, dev_image_input); //Sets input as the input buffer
		histKernel.setArg(1, histogramBuffer); //And output as the histogram buffer
		//histKernel.setArg(2, 64);
		//histKernel.setArg(3, 256);
		queue.enqueueNDRangeKernel(histKernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &prof_event);//Executes the histKernel
		queue.enqueueReadBuffer(histogramBuffer, CL_TRUE, 0, outputSize, &histogramVector.data()[0]); //Reads from the histogram buffer
		std::cout << histogramVector << std::endl; //Outputs the vector

		totalExecutionTime = executionTime(totalExecutionTime, prof_event);

		

		//---Scan Add---
		queue.enqueueWriteBuffer(histogramBuffer, CL_TRUE, 0, histogramVector.size(), &histogramVector.data()[0]); //Queues the historgram buffer as a write buffer 
		queue.enqueueFillBuffer(scanAddBuffer, 0, 0, outputSize); //Fills using the scanAddBuffer
		cl::Kernel scanAddKernel = cl::Kernel(program, "scan_add"); //Creates a scan add kernel
		scanAddKernel.setArg(0, histogramBuffer); //Inputs the histogram buffer
		scanAddKernel.setArg(1, scanAddBuffer); //Outputs with the scan add buffer
		scanAddKernel.setArg(2, cl::Local(localSize * sizeof(type))); //local memory size
		scanAddKernel.setArg(3, cl::Local(localSize * sizeof(type)));


		queue.enqueueNDRangeKernel(scanAddKernel, cl::NullRange, cl::NDRange(scanAddVector.size()), cl::NullRange, NULL, &prof_event);//Executes the scanAddKernel
		queue.enqueueReadBuffer(scanAddBuffer, CL_TRUE, 0, outputSize, &scanAddVector.data()[0]); //Reads from the scanAddBuffer
		std::cout << scanAddVector << std::endl; //Outputs the scanAddVector

		totalExecutionTime = executionTime(totalExecutionTime, prof_event);

		//---BlockSum---
		queue.enqueueWriteBuffer(scanAddBuffer, CL_TRUE, 0, scanAddVector.size(), &scanAddVector.data()[0]); //Writes to the scanAddBuffer 
		queue.enqueueFillBuffer(blockSumBuffer, 0, 0, outputSize); //Fills using the blockSumBuffer
		cl::Kernel blockSumKernel = cl::Kernel(program, "block_sum"); //Creates a kernel instance of block_sum
		blockSumKernel.setArg(0, scanAddBuffer); //Passes the scan add buffer and the block sum buffer
		blockSumKernel.setArg(1, blockSumBuffer);
		blockSumKernel.setArg(2, 32); //Uses the local size variable of 32 


		
		queue.enqueueNDRangeKernel(blockSumKernel, cl::NullRange, cl::NDRange(blockSumVector.size()), cl::NullRange, NULL, &prof_event);
		queue.enqueueReadBuffer(blockSumBuffer, CL_TRUE, 0, outputSize, &blockSumVector.data()[0]); //Reads from the block sum buffer into the block sum vector
		std::cout << blockSumVector << std::endl; //Ouputs the block sum vector

		totalExecutionTime = executionTime(totalExecutionTime, prof_event);

		//---Scan Add Atomic---
		queue.enqueueWriteBuffer(blockSumBuffer, CL_TRUE, 0, blockSumVector.size(), &blockSumVector.data()[0]); //Writes to the block sum buffer with the block sum vector
		queue.enqueueFillBuffer(scanAddAtomicBuffer, 0, 0, outputSize); //Fills the scan add atomic buffer
		cl::Kernel scanAddAtomicKernel = cl::Kernel(program, "scan_add_atomic"); //Creates a kernel instance of scan add atomic kernel
		scanAddAtomicKernel.setArg(0, blockSumBuffer); //Passes the block sum buffer as input and outputs to the scan add atomic buffer
		scanAddAtomicKernel.setArg(1, scanAddAtomicBuffer);

		queue.enqueueNDRangeKernel(scanAddAtomicKernel, cl::NullRange, cl::NDRange(scanAddAtomicVector.size()), cl::NullRange, NULL, &prof_event); //Executes the kernel keeping track of the event
		queue.enqueueReadBuffer(scanAddAtomicBuffer, CL_TRUE, 0, outputSize, &scanAddAtomicVector.data()[0]);
		std::cout << scanAddAtomicVector << std::endl;

		totalExecutionTime = executionTime(totalExecutionTime, prof_event);

		//---Scan Add Adjust---
		//Writes to the scan add atomic buffer passing the scan add atomic vector data and size then writes to the scan add buffer
		queue.enqueueWriteBuffer(scanAddAtomicBuffer, CL_TRUE, 0, scanAddAtomicVector.size(), &scanAddAtomicVector.data()[0]);
		queue.enqueueWriteBuffer(scanAddBuffer, CL_TRUE, 0, scanAddVector.size(), &scanAddVector.data()[0]);

		cl::Kernel scanAddAdjustKernel = cl::Kernel(program, "scan_add_adjust"); //Creates a scan add adjust kernel
		scanAddAdjustKernel.setArg(0, scanAddAtomicBuffer); 
		scanAddAdjustKernel.setArg(1, scanAddBuffer); //Outputs into the scan add buffer


		queue.enqueueNDRangeKernel(scanAddAdjustKernel, cl::NullRange, cl::NDRange(scanAddAdjustVector.size()),cl::NullRange, NULL, &prof_event);
		queue.enqueueReadBuffer(scanAddBuffer, CL_TRUE, 0, outputSize, &scanAddAdjustVector.data()[0]);//Reads from the scan add buffer 
		std::cout << scanAddAdjustVector << std::endl;

		totalExecutionTime = executionTime(totalExecutionTime, prof_event);

		//---Normalisation---
		//Writes to the scan add buffer
		queue.enqueueWriteBuffer(scanAddBuffer, CL_TRUE, 0, scanAddVector.size(), &scanAddVector.data()[0]);
		queue.enqueueFillBuffer(normaliseAndScale, 0, 0, outputSize); //fills the normalise and scale buffer
		cl::Kernel normalisedKernel = cl::Kernel(program, "normalise_and_scale"); //creates and instance of the normalise and scale kernel
		normalisedKernel.setArg(0, scanAddBuffer); //Uses the scan add buffer and normalises the data by dividing by 2732
		normalisedKernel.setArg(1, scanAddBuffer);
		normalisedKernel.setArg(2, 2732);


		queue.enqueueNDRangeKernel(normalisedKernel, cl::NullRange, cl::NDRange(normalisedVector.size()), cl::NullRange, NULL, &prof_event);
		queue.enqueueReadBuffer(scanAddBuffer, CL_TRUE, 0, outputSize, &normalisedVector.data()[0]); //Reads from the scan add buffer to produce the normalised vector
		std::cout << normalisedVector << std::endl;


		totalExecutionTime = executionTime(totalExecutionTime, prof_event);

		//---Back Projection---
		queue.enqueueWriteBuffer(normaliseAndScale, CL_TRUE, 0, normalisedVector.size(), &normalisedVector.data()[0]);//Writes using the normalise and scale buffer
		queue.enqueueFillBuffer(normaliseAndScale, 0, 0, outputSize);//Fills the normalise and scale buffer
		cl::Kernel backProjectionKernel = cl::Kernel(program, "back_projection");//Creates an instance of the back_projection kernel
		backProjectionKernel.setArg(0, scanAddBuffer); //Uses the scan add buffer as the lookup table
		backProjectionKernel.setArg(1, dev_image_input); //Passes the input and output image
		backProjectionKernel.setArg(2, dev_image_output);
		queue.enqueueNDRangeKernel(backProjectionKernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &prof_event);
		queue.enqueueReadBuffer(scanAddBuffer, CL_TRUE, 0, outputSize, &image_input.data()[0]); //Reads from the scan add buffer 
		std::cout << normalisedVector << std::endl;

		totalExecutionTime = executionTime(totalExecutionTime, prof_event);

		std::cout << "Total Kernel execution time [ns]:" << totalExecutionTime << endl;



		vector<unsigned char> output_buffer(image_input.size()); //Initialise output buffer with the size of the image input
		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]); //Reads from the image output buffer into the final output buffer

		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum()); //Creates the output image using the output buffer data
		CImgDisplay disp_output(output_image,"output"); //Displays the output image


 		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
		    disp_input.wait(1);
		    disp_output.wait(1);
	    }		

	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
