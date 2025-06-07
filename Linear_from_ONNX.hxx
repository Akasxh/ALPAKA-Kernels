#include <alpaka/alpaka.hpp>

#include <cublas_v2.h>
#include <cuda.h>

#include <algorithm>
#include <vector>


using Dim1D = alpaka::DimInt<1>;

template <EAccType>
struct AccFromEnum;

template <>
struct AccFromEnum<EAccType::CUDA> {
    using Type = alpaka::TagToAcc<alpaka::TagGpuCudaRt, alpaka::DimInt<1>, std::size_t>;
};



namespace SOFIE_Linear_4{

    template <EAccType Acc>
    struct Session {
    // initialized tensors
    float tensor_B[3] = { -0.312488616, 0.409787357, 0.970163822};
    float tensor_W[12] = { 2.30650759, 0.30019927, 0.714929163, 0.770087719, -0.000216799875, 1.24266589, -0.27147606, -0.0746395588, 0.719775498, -1.55395639, -2.31602883, -0.204315066};
    
    using Idx = std::size_t;
    using DataType = float;
    using Dim = alpaka::DimInt<1>;
    using AccType = typename AccFromEnum<Acc>::Type;
    using Queue = alpaka::Queue<AccType, alpaka::Blocking>;

    // Declare platform and device for host
    alpaka::PlatformCpu const platformHost{};
    alpaka::DevCpu const devHost = alpaka::getDevByIdx(platformHost, 0);
    
    // Declare platform and device for accelerator
    alpaka::Platform<AccType> const platformAcc{};
    alpaka::Dev<AccType> const devAcc = alpaka::getDevByIdx(platformAcc, 0);

    // Initialize the queue
    Queue queue; 

    // ALPAKA Initialized Host Buffers
    alpaka::Buf<alpaka::DevCpu, DataType, Dim, Idx> weight_buffer_host;
    alpaka::Buf<alpaka::DevCpu, DataType, Dim, Idx> bias_buffer_host;
    alpaka::Buf<AccType, DataType, Dim, Idx> weight_buffer_dev;
    alpaka::Buf<AccType, DataType, Dim, Idx> bias_buffer_dev;
    alpaka::Buf<AccType, DataType, Dim, Idx> output_buffer_dev;

    // Initializing SOFIE BLAS Backend
    SOFIE::BLASBackend<EHetType::ALPAKA, Acc> SOFIE_ALPAKA_BLAS;
    
    Session(): queue(devAcc) {
        // Allocate host buffers
        weight_buffer_host = alpaka::allocBuf<DataType, Idx>(devHost, 12);
        bias_buffer_host = alpaka::allocBuf<DataType, Idx>(devHost, 3);

        // Copy data to host buffers
        alpaka::memcpy(devHost, weight_buffer_host, tensor_W, sizeof(DataType) * 12);
        alpaka::memcpy(devHost, bias_buffer_host, tensor_B, sizeof(DataType) * 3);

        // Allocate and copy to device buffers
        weight_buffer_dev = alpaka::allocBuf<DataType, Idx>(devAcc, 12);
        bias_buffer_dev = alpaka::allocBuf<DataType, Idx>(devAcc, 3);
        alpaka::memcpy(queue, weight_buffer_dev, weight_buffer_host);
        alpaka::memcpy(queue, bias_buffer_dev, bias_buffer_host);

        // Allocate output buffer
        output_buffer_dev = alpaka::allocBuf<DataType, Idx>(devAcc, 3);

        alpaka::wait(queue);
    }
    
    
    alpaka::Buf<AccType, DataType, Dim, Idx>
    infer_alpaka(alpaka::Buf<AccType, DataType, Dim, Idx> &tensor_input){
    
    //--------- Gemm
       char op_0_transA = 'n';
       char op_0_transB = 'n';
       int op_0_m = 1;
       int op_0_n = 3;
       int op_0_k = 4;
       float op_0_alpha = 1;
       float op_0_beta = 1;
       int op_0_lda = 4;
       int op_0_ldb = 3;
    
       alpaka::memcpy(queue, output_buffer_dev, bias_buffer_dev);
       SOFIE_ALPAKA_BLAS.gemm(queue, &op_0_transA, &op_0_transB, &op_0_m, &op_0_n, &op_0_k, &op_0_alpha,
                                weight_buffer_dev, &op_0_lda, tensor_input, &op_0_ldb, &op_0_beta, output_buffer_dev, &op_0_n);

       alpaka::wait(queue);
       return output_buffer_dev;
    }
    }; // end of Session
    } //SOFIE_Linear_4
