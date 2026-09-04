#include <stdio.h>
#include <stdbool.h>
#include <nvml.h>

bool initNVML(nvmlDevice_t *device) {
    nvmlReturn_t result = nvmlInit();
    if (NVML_SUCCESS != result) {
        fprintf(stderr, "[!] Errore inizializzazione NVML: %s\n", nvmlErrorString(result));
        return false;
    }
    
    result = nvmlDeviceGetHandleByIndex(0, device);
    if (NVML_SUCCESS != result) {
        fprintf(stderr, "[!] Errore recupero handle GPU 0: %s\n", nvmlErrorString(result));
        nvmlShutdown();
        return false;
    }
    return true;
}

void printGPUStats(nvmlDevice_t device) {
    unsigned int temp = 0;
    
    // Struttura di temperatura NVML V1
    nvmlTemperature_t tempStruct;
    tempStruct.version = nvmlTemperature_v1;
    tempStruct.sensorType = NVML_TEMPERATURE_GPU;

    // Chiama la nuova API V1 senza warning
    if (nvmlDeviceGetTemperatureV(device, &tempStruct) == NVML_SUCCESS) {
        temp = tempStruct.temperature; // Il campo si chiama "temperature", non "value"
    }

    nvmlUtilization_t utilization;
    nvmlMemory_t memory;
    
    nvmlDeviceGetUtilizationRates(device, &utilization);
    nvmlDeviceGetMemoryInfo(device, &memory);

    printf("[GPU Stats] Utilizzo Core: %u%% | Temperatura: %u°C | VRAM: %llu MB / %llu MB\n", 
           utilization.gpu, 
           temp, 
           (unsigned long long)(memory.used / (1024 * 1024)), 
           (unsigned long long)(memory.total / (1024 * 1024)));
}