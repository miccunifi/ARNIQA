#include <stdio.h>
#include <vector>
#include <omp.h>
#include <math.h>
#include <algorithm>

typedef struct {
    float R, G, B;
} DitherStruct;

typedef struct {
    int index;
} DitherStructDiff;

void FindNearestColor(float *color, float *ptr_pal, int &bestIndex, int nc) {
    float minDistanceSquared = __FLT_MAX__;
    int r_idx_p = nc*0;
    int g_idx_p = nc*1;
    int b_idx_p = nc*2;

    for (int i = 0; i < nc; i++) {
        float Rdiff = color[0] - ptr_pal[r_idx_p + i];
        float Gdiff = color[1] - ptr_pal[g_idx_p + i];
        float Bdiff = color[2] - ptr_pal[b_idx_p + i];
        float distanceSquared = Rdiff * Rdiff + Gdiff * Gdiff + Bdiff * Bdiff;

        if (distanceSquared < minDistanceSquared) {
            minDistanceSquared = distanceSquared;
            bestIndex= i;
        }
        
    }
}


extern "C" {
void dither(float* input_t, float* p_t, int w, int h, int nc) {
    
    float a = 7./16.;
    float b = 3./16.;
    float c = 5./16.;
    float d = 1./16.;
    float quant_error_R = 0.;
    float quant_error_G = 0.;
    float quant_error_B = 0.;
    float old_pixel[3];
    float min_v = 0.;
    float max_v = 1.;
    
    int bestIndex = -1;
    
    int r_idx_in = h*w*0;
    int g_idx_in = h*w*1;
    int b_idx_in = h*w*2;
    int r_idx_p = nc*0;
    int g_idx_p = nc*1;
    int b_idx_p = nc*2;


    for (int y = 0; y < h-1; y++) {
        for (int x = 1; x < w-1; x++) {

            old_pixel[0] = input_t[r_idx_in + w*y + x];
            old_pixel[1] = input_t[g_idx_in + w*y + x];
            old_pixel[2] = input_t[b_idx_in + w*y + x];

            FindNearestColor(old_pixel, p_t, bestIndex, nc);

            input_t[r_idx_in + w*y + x] = p_t[r_idx_p + bestIndex];
            input_t[g_idx_in + w*y + x] = p_t[g_idx_p + bestIndex];
            input_t[b_idx_in + w*y + x] = p_t[b_idx_p + bestIndex];

            quant_error_R = old_pixel[0] -  input_t[r_idx_in + w*y + x];
            quant_error_G = old_pixel[1] -  input_t[g_idx_in + w*y + x];
            quant_error_B = old_pixel[2] -  input_t[b_idx_in + w*y + x];

            input_t[r_idx_in + w*y     + (x+1)] = std::clamp(input_t[r_idx_in + w*y     + (x+1)] + quant_error_R * a, min_v, max_v);
            input_t[g_idx_in + w*y     + (x+1)] = std::clamp(input_t[g_idx_in + w*y     + (x+1)] + quant_error_G * a, min_v, max_v);
            input_t[b_idx_in + w*y     + (x+1)] = std::clamp(input_t[b_idx_in + w*y     + (x+1)] + quant_error_B * a, min_v, max_v);

            input_t[r_idx_in + w*(y+1) + (x-1)] = std::clamp(input_t[r_idx_in + w*(y+1) + (x-1)] + quant_error_R * b, min_v, max_v);
            input_t[g_idx_in + w*(y+1) + (x-1)] = std::clamp(input_t[g_idx_in + w*(y+1) + (x-1)] + quant_error_G * b, min_v, max_v);
            input_t[b_idx_in + w*(y+1) + (x-1)] = std::clamp(input_t[b_idx_in + w*(y+1) + (x-1)] + quant_error_B * b, min_v, max_v);

            input_t[r_idx_in + w*(y+1) +  x   ] = std::clamp(input_t[r_idx_in + w*(y+1) +  x   ] + quant_error_R * c, min_v, max_v);
            input_t[g_idx_in + w*(y+1) +  x   ] = std::clamp(input_t[g_idx_in + w*(y+1) +  x   ] + quant_error_G * c, min_v, max_v);
            input_t[b_idx_in + w*(y+1) +  x   ] = std::clamp(input_t[b_idx_in + w*(y+1) +  x   ] + quant_error_B * c, min_v, max_v);

            input_t[r_idx_in + w*(y+1) + (x+1)] = std::clamp(input_t[r_idx_in + w*(y+1) + (x+1)] + quant_error_R * d, min_v, max_v);
            input_t[g_idx_in + w*(y+1) + (x+1)] = std::clamp(input_t[g_idx_in + w*(y+1) + (x+1)] + quant_error_G * d, min_v, max_v);
            input_t[b_idx_in + w*(y+1) + (x+1)] = std::clamp(input_t[b_idx_in + w*(y+1) + (x+1)] + quant_error_B * d, min_v, max_v);
            
        }
    }
}
}