/*
 * Copyright (c) 2024 AlexD Oleksandr Don
 * 
 * Power System State Estimation with CUDA
 * 
 * DLL Export/Import Macros
 */

#ifndef SLE_EXPORT_H
#define SLE_EXPORT_H

#ifdef _WIN32
    #ifdef SLE_BUILD_SHARED_LIBS
        #ifdef SLE_EXPORTS
            #define SLE_API __declspec(dllexport)
        #else
            #define SLE_API __declspec(dllimport)
        #endif
    #else
        #define SLE_API
    #endif
#else
    #if __GNUC__ >= 4
        #define SLE_API __attribute__ ((visibility ("default")))
    #else
        #define SLE_API
    #endif
#endif

#endif // SLE_EXPORT_H

