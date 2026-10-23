// Streaming read bandwidth of the GPU with sub-group block reads. On Pantherlake-P this saturates at ~42 GB/s for
// buffers from 16 MB to 512 MB (and ~34 GB/s for 2-6 MB), i.e. there is no useful cache between kernels, which is why
// the elementwise kernels of a prefill all sit at the DRAM floor.
//   gcc dram_read_bw.c -o dram_read_bw -lOpenCL && MB=256 ./dram_read_bw
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
static const char* src =
"#pragma OPENCL EXTENSION cl_intel_subgroups : enable\n"
"__attribute__((intel_reqd_sub_group_size(16))) __kernel void rdb(__global const uint* a, __global uint* o, uint nvec, uint per){ uint sg=get_group_id(0)*(get_local_size(0)/16)+get_sub_group_id(); uint s=0; size_t base=(size_t)sg*per*16*4; for(uint i=0;i<per;i+=4){ uint4 x0=intel_sub_group_block_read4(a+base+(size_t)(i+0)*64); uint4 x1=intel_sub_group_block_read4(a+base+(size_t)(i+1)*64); uint4 x2=intel_sub_group_block_read4(a+base+(size_t)(i+2)*64); uint4 x3=intel_sub_group_block_read4(a+base+(size_t)(i+3)*64); uint4 y=x0+x1+x2+x3; s+=y.x+y.y+y.z+y.w;} if(s==0xdeadbeef) o[0]=s; }\n";
int main(int argc,char**argv){
  cl_platform_id p[4]; cl_uint np; clGetPlatformIDs(4,p,&np); cl_device_id d; cl_int e;
  for(int i=0;i<np;i++) if(clGetDeviceIDs(p[i],CL_DEVICE_TYPE_GPU,1,&d,NULL)==CL_SUCCESS) break;
  cl_context c=clCreateContext(NULL,1,&d,NULL,NULL,&e);
  cl_command_queue q=clCreateCommandQueueWithProperties(c,d,(cl_queue_properties[]){CL_QUEUE_PROPERTIES,CL_QUEUE_PROFILING_ENABLE,0},&e);
  cl_program pr=clCreateProgramWithSource(c,1,&src,NULL,&e); e=clBuildProgram(pr,1,&d,"",NULL,NULL); if(e){char l[4096];clGetProgramBuildInfo(pr,d,CL_PROGRAM_BUILD_LOG,4096,l,NULL);puts(l);return 1;}
  size_t bytes=(size_t)atoi(getenv("MB"))<<20; 
  cl_mem a=clCreateBuffer(c,CL_MEM_READ_WRITE,bytes,NULL,&e), o=clCreateBuffer(c,CL_MEM_READ_WRITE,64,NULL,&e);
  void* h=malloc(bytes); for(size_t i=0;i<bytes/4;i++) ((unsigned*)h)[i]=rand(); clEnqueueWriteBuffer(q,a,1,0,bytes,h,0,NULL,NULL);
  cl_kernel kn=clCreateKernel(pr,"rdb",&e);
  for(int nsg_total=2048; nsg_total<=65536; nsg_total*=2){
    unsigned per=(bytes/ (16*16)) / nsg_total; per&=~3u; if(per<4) continue; unsigned nvec=0;
    clSetKernelArg(kn,0,8,&a);clSetKernelArg(kn,1,8,&o);clSetKernelArg(kn,2,4,&nvec);clSetKernelArg(kn,3,4,&per);
    size_t lws=16*8, gws=(size_t)nsg_total*16; double best=1e9;
    for(int it=0;it<8;it++){ cl_event ev; e=clEnqueueNDRangeKernel(q,kn,1,NULL,&gws,&lws,0,NULL,&ev); clWaitForEvents(1,&ev);
      cl_ulong s,t; clGetEventProfilingInfo(ev,CL_PROFILING_COMMAND_START,8,&s,NULL); clGetEventProfilingInfo(ev,CL_PROFILING_COMMAND_END,8,&t,NULL); double ms=(t-s)/1e6; if(ms<best)best=ms; }
    double rb=(double)per*nsg_total*256; printf("sgs=%d per=%u: %.2f ms %.1f GB/s\n",nsg_total,per,best,rb/best/1e6);
  }
}
