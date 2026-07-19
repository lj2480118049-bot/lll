// ============================================================
// AI Student Performance Analysis System - English Version
// For Bear C / Raylib single-file environment
// ============================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <raylib.h>

// ============================================================
// Constants & Structs
// ============================================================

#define MAX_STUDENTS 2000
#define MAX_LINE 1024
#define FEATURE_DIM 4

typedef enum {
    MODE_LINEAR_REGRESSION = 0,
    MODE_KMEANS = 1,
    MODE_KNN = 2
} AppMode;

typedef struct Student {
    int student_id;
    int major;
    int year;
    float gpa_start;
    float ai_hour;
    int ai_purpose;
    float gpa_end;
    float memory;
    int cluster;
} Student;

typedef struct NormalizeParams {
    float age_min, age_max;
    float gpa_min, gpa_max;
    float hour_min, hour_max;
    float purpose_min, purpose_max;
    float memory_min, memory_max;
} NormalizeParams;

typedef struct LinearModel {
    float w_age, w_gpa, w_hour, w_purpose;
    float bias;
    float learning_rate;
    int iterations;
    float *loss_history;
    int loss_count;
} LinearModel;

typedef struct KMeansModel {
    int k;
    float centroids[10][FEATURE_DIM];
    float sse;
    int cluster_counts[10];
} KMeansModel;

typedef struct KNNModel {
    int k;
    Student *train_data;
    int train_count;
} KNNModel;

typedef struct Dataset {
    Student *train;
    int train_count;
    Student *test;
    int test_count;
    NormalizeParams norm;
} Dataset;

typedef struct AppState {
    Dataset dataset;
    LinearModel lr_model;
    KMeansModel km_model;
    KNNModel knn_model;
    AppMode current_mode;
    
    int input_year;
    float input_gpa;
    float input_hour;
    int input_purpose;
    int knn_predicted_cluster;
    float knn_predicted_gpa;
    
    int kmeans_iteration;
    int kmeans_training;
    
    int screen_width;
    int screen_height;
} AppState;

// ============================================================
// Function Declarations
// ============================================================

int encode_major(const char *major);
int encode_year(const char *year);
int encode_ai_purpose(const char *purpose);
int load_csv(const char *filename, Student *students, int max_count);
void compute_normalize_params(Student *students, int count, NormalizeParams *params);
void normalize_student(const Student *s, const NormalizeParams *norm, float *features);
void split_dataset(Student *all_students, int total_count,
                   Student **train, int *train_count,
                   Student **test, int *test_count, float train_ratio);
void export_clusters_csv(const char *filename, Student *students, int count);
void export_predictions_csv(const char *filename, Student *students, int count, float *predictions);

void linear_model_init(LinearModel *model, float learning_rate, int max_iterations);
void linear_train(LinearModel *model, Student *train_data, int train_count, NormalizeParams *norm);
float linear_predict(LinearModel *model, const Student *s, NormalizeParams *norm);
float linear_evaluate(LinearModel *model, Student *test_data, int test_count, NormalizeParams *norm);
void linear_model_free(LinearModel *model);

void kmeans_init(KMeansModel *model, int k);
void kmeans_random_centroids(KMeansModel *model, Student *data, int count, NormalizeParams *norm);
int kmeans_step(KMeansModel *model, Student *data, int count, NormalizeParams *norm);
void kmeans_train(KMeansModel *model, Student *data, int count, NormalizeParams *norm);
float kmeans_compute_sse(KMeansModel *model, Student *data, int count, NormalizeParams *norm);
int kmeans_predict(KMeansModel *model, const Student *s, NormalizeParams *norm);

void knn_init(KNNModel *model, int k, Student *train_data, int train_count);
float knn_distance(const float *a, const float *b, int dim);
int knn_predict_cluster(KNNModel *model, const Student *s, NormalizeParams *norm);
float knn_predict_gpa(KNNModel *model, const Student *s, NormalizeParams *norm);
float knn_evaluate_accuracy(KNNModel *model, Student *test_data, int test_count, NormalizeParams *norm);

void gui_init(AppState *state, int width, int height);
void gui_handle_input(AppState *state);
void gui_render(AppState *state);
void gui_cleanup(AppState *state);

// ============================================================
// Data Module
// ============================================================

int encode_major(const char *major) {
    if (strcmp(major, "STEM") == 0) return 1;
    if (strcmp(major, "Humanities") == 0) return 2;
    if (strcmp(major, "Arts") == 0) return 3;
    if (strcmp(major, "Business") == 0) return 4;
    if (strcmp(major, "Medical") == 0) return 5;
    return 0;
}

int encode_year(const char *year) {
    if (strcmp(year, "Freshman") == 0) return 1;
    if (strcmp(year, "Sophomore") == 0) return 2;
    if (strcmp(year, "Junior") == 0) return 3;
    if (strcmp(year, "Senior") == 0) return 4;
    if (strcmp(year, "Graduate") == 0) return 5;
    return 0;
}

int encode_ai_purpose(const char *purpose) {
    if (strcmp(purpose, "Summarizing_Reading") == 0) return 0;
    if (strcmp(purpose, "Ideation") == 0) return 1;
    if (strcmp(purpose, "Debugging/Troubleshooting") == 0) return 2;
    if (strcmp(purpose, "Copywriting/Drafting") == 0) return 3;
    if (strcmp(purpose, "Direct_Answer_Generation") == 0) return 4;
    return 0;
}

static void csv_split(char *line, char **fields, int max_fields) {
    int i = 0, field = 0;
    int in_quotes = 0;
    fields[field] = line;
    
    while (line[i] && field < max_fields - 1) {
        if (line[i] == '"') {
            in_quotes = !in_quotes;
        } else if (line[i] == ',' && !in_quotes) {
            line[i] = '\0';
            field++;
            fields[field] = &line[i + 1];
        }
        i++;
    }
}

int load_csv(const char *filename, Student *students, int max_count) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        printf("Error: Cannot open file %s\n", filename);
        return -1;
    }
    
    char line[MAX_LINE];
    char *fields[10];
    int count = 0;
    
    if (fgets(line, MAX_LINE, fp) == NULL) {
        fclose(fp);
        return 0;
    }
    
    while (fgets(line, MAX_LINE, fp) && count < max_count) {
        line[strcspn(line, "\r\n")] = 0;
        if (strlen(line) == 0) continue;
        
        csv_split(line, fields, 8);
        
        students[count].student_id = atoi(fields[0]);
        students[count].major = encode_major(fields[1]);
        students[count].year = encode_year(fields[2]);
        students[count].gpa_start = atof(fields[3]);
        students[count].ai_hour = atof(fields[4]);
        students[count].ai_purpose = encode_ai_purpose(fields[5]);
        students[count].gpa_end = atof(fields[6]);
        students[count].memory = atof(fields[7]);
        students[count].cluster = -1;
        
        count++;
    }
    
    fclose(fp);
    printf("Loaded %d students from %s\n", count, filename);
    return count;
}

void compute_normalize_params(Student *students, int count, NormalizeParams *params) {
    if (count == 0) return;
    
    params->age_min = params->age_max = students[0].year;
    params->gpa_min = params->gpa_max = students[0].gpa_start;
    params->hour_min = params->hour_max = students[0].ai_hour;
    params->purpose_min = params->purpose_max = students[0].ai_purpose;
    params->memory_min = params->memory_max = students[0].memory;
    
    for (int i = 1; i < count; i++) {
        if (students[i].year < params->age_min) params->age_min = students[i].year;
        if (students[i].year > params->age_max) params->age_max = students[i].year;
        
        if (students[i].gpa_start < params->gpa_min) params->gpa_min = students[i].gpa_start;
        if (students[i].gpa_start > params->gpa_max) params->gpa_max = students[i].gpa_start;
        
        if (students[i].ai_hour < params->hour_min) params->hour_min = students[i].ai_hour;
        if (students[i].ai_hour > params->hour_max) params->hour_max = students[i].ai_hour;
        
        if (students[i].ai_purpose < params->purpose_min) params->purpose_min = students[i].ai_purpose;
        if (students[i].ai_purpose > params->purpose_max) params->purpose_max = students[i].ai_purpose;
        
        if (students[i].memory < params->memory_min) params->memory_min = students[i].memory;
        if (students[i].memory > params->memory_max) params->memory_max = students[i].memory;
    }
}

void normalize_student(const Student *s, const NormalizeParams *norm, float *features) {
    float age_range = norm->age_max - norm->age_min;
    float gpa_range = norm->gpa_max - norm->gpa_min;
    float hour_range = norm->hour_max - norm->hour_min;
    float purpose_range = norm->purpose_max - norm->purpose_min;
    
    features[0] = age_range > 0 ? (s->year - norm->age_min) / age_range : 0.5f;
    features[1] = gpa_range > 0 ? (s->gpa_start - norm->gpa_min) / gpa_range : 0.5f;
    features[2] = hour_range > 0 ? (s->ai_hour - norm->hour_min) / hour_range : 0.5f;
    features[3] = purpose_range > 0 ? (s->ai_purpose - norm->purpose_min) / purpose_range : 0.5f;
}

void split_dataset(Student *all_students, int total_count,
                   Student **train, int *train_count,
                   Student **test, int *test_count, float train_ratio) {
    srand(42);
    for (int i = total_count - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        Student temp = all_students[i];
        all_students[i] = all_students[j];
        all_students[j] = temp;
    }
    
    *train_count = (int)(total_count * train_ratio);
    *test_count = total_count - *train_count;
    
    *train = (Student *)malloc(sizeof(Student) * (*train_count));
    *test = (Student *)malloc(sizeof(Student) * (*test_count));
    
    memcpy(*train, all_students, sizeof(Student) * (*train_count));
    memcpy(*test, all_students + *train_count, sizeof(Student) * (*test_count));
    
    printf("Dataset split: %d train, %d test\n", *train_count, *test_count);
}

void export_clusters_csv(const char *filename, Student *students, int count) {
    FILE *fp = fopen(filename, "w");
    if (!fp) return;
    
    fprintf(fp, "Student_ID,Major,Year,GPA_Start,AI_Hours,AI_Purpose,GPA_End,Memory,Cluster\n");
    for (int i = 0; i < count; i++) {
        fprintf(fp, "%d,%d,%d,%.3f,%.2f,%d,%.3f,%.2f,%d\n",
                students[i].student_id, students[i].major, students[i].year,
                students[i].gpa_start, students[i].ai_hour, students[i].ai_purpose,
                students[i].gpa_end, students[i].memory, students[i].cluster);
    }
    fclose(fp);
    printf("Cluster results exported to %s\n", filename);
}

void export_predictions_csv(const char *filename, Student *students, int count, float *predictions) {
    FILE *fp = fopen(filename, "w");
    if (!fp) return;
    
    fprintf(fp, "Student_ID,Actual_GPA,Predicted_GPA,Error\n");
    for (int i = 0; i < count; i++) {
        float error = students[i].gpa_end - predictions[i];
        fprintf(fp, "%d,%.3f,%.3f,%.3f\n",
                students[i].student_id, students[i].gpa_end, predictions[i], error);
    }
    fclose(fp);
    printf("Prediction results exported to %s\n", filename);
}

// ============================================================
// Algorithm Module
// ============================================================

// --- Linear Regression ---

void linear_model_init(LinearModel *model, float learning_rate, int max_iterations) {
    model->w_age = 0;
    model->w_gpa = 0;
    model->w_hour = 0;
    model->w_purpose = 0;
    model->bias = 0;
    model->learning_rate = learning_rate;
    model->iterations = max_iterations;
    model->loss_history = (float *)malloc(sizeof(float) * max_iterations);
    model->loss_count = 0;
}

void linear_train(LinearModel *model, Student *train_data, int train_count, NormalizeParams *norm) {
    int n = train_count;
    model->loss_count = 0;
    
    for (int iter = 0; iter < model->iterations; iter++) {
        float grad_w0 = 0, grad_w1 = 0, grad_w2 = 0, grad_w3 = 0;
        float grad_b = 0;
        float total_loss = 0;
        
        for (int i = 0; i < n; i++) {
            float features[FEATURE_DIM];
            normalize_student(&train_data[i], norm, features);
            
            float pred = model->w_age * features[0] + model->w_gpa * features[1]
                        + model->w_hour * features[2] + model->w_purpose * features[3]
                        + model->bias;
            
            float error = pred - train_data[i].gpa_end;
            total_loss += error * error;
            
            grad_w0 += error * features[0];
            grad_w1 += error * features[1];
            grad_w2 += error * features[2];
            grad_w3 += error * features[3];
            grad_b += error;
        }
        
        float lr = model->learning_rate;
        model->w_age -= lr * (2.0f / n) * grad_w0;
        model->w_gpa -= lr * (2.0f / n) * grad_w1;
        model->w_hour -= lr * (2.0f / n) * grad_w2;
        model->w_purpose -= lr * (2.0f / n) * grad_w3;
        model->bias -= lr * (2.0f / n) * grad_b;
        
        model->loss_history[model->loss_count++] = total_loss / n;
    }
    
    printf("Linear Regression trained. Final MSE: %.4f\n", model->loss_history[model->loss_count - 1]);
}

float linear_predict(LinearModel *model, const Student *s, NormalizeParams *norm) {
    float features[FEATURE_DIM];
    normalize_student(s, norm, features);
    
    return model->w_age * features[0] + model->w_gpa * features[1]
           + model->w_hour * features[2] + model->w_purpose * features[3]
           + model->bias;
}

float linear_evaluate(LinearModel *model, Student *test_data, int test_count, NormalizeParams *norm) {
    float total_loss = 0;
    for (int i = 0; i < test_count; i++) {
        float pred = linear_predict(model, &test_data[i], norm);
        float error = pred - test_data[i].gpa_end;
        total_loss += error * error;
    }
    return total_loss / test_count;
}

void linear_model_free(LinearModel *model) {
    if (model->loss_history) {
        free(model->loss_history);
        model->loss_history = NULL;
    }
}

// --- KMeans ---

void kmeans_init(KMeansModel *model, int k) {
    model->k = k;
    model->sse = 0;
    for (int i = 0; i < k; i++) {
        model->cluster_counts[i] = 0;
        for (int j = 0; j < FEATURE_DIM; j++) {
            model->centroids[i][j] = 0;
        }
    }
}

void kmeans_random_centroids(KMeansModel *model, Student *data, int count, NormalizeParams *norm) {
    srand(123);
    for (int i = 0; i < model->k; i++) {
        int idx = rand() % count;
        float features[FEATURE_DIM];
        normalize_student(&data[idx], norm, features);
        for (int j = 0; j < FEATURE_DIM; j++) {
            model->centroids[i][j] = features[j];
        }
    }
}

static int find_nearest_cluster(const float *features, KMeansModel *model) {
    int best = 0;
    float min_dist = 1e9;
    
    for (int c = 0; c < model->k; c++) {
        float dist = 0;
        for (int d = 0; d < FEATURE_DIM; d++) {
            float diff = features[d] - model->centroids[c][d];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            best = c;
        }
    }
    return best;
}

int kmeans_step(KMeansModel *model, Student *data, int count, NormalizeParams *norm) {
    float new_centroids[10][FEATURE_DIM] = {0};
    int counts[10] = {0};
    
    for (int i = 0; i < count; i++) {
        float features[FEATURE_DIM];
        normalize_student(&data[i], norm, features);
        int c = find_nearest_cluster(features, model);
        data[i].cluster = c;
        counts[c]++;
        for (int d = 0; d < FEATURE_DIM; d++) {
            new_centroids[c][d] += features[d];
        }
    }
    
    float max_shift = 0;
    for (int c = 0; c < model->k; c++) {
        model->cluster_counts[c] = counts[c];
        if (counts[c] > 0) {
            for (int d = 0; d < FEATURE_DIM; d++) {
                new_centroids[c][d] /= counts[c];
                float shift = fabsf(new_centroids[c][d] - model->centroids[c][d]);
                if (shift > max_shift) max_shift = shift;
                model->centroids[c][d] = new_centroids[c][d];
            }
        }
    }
    
    return max_shift < 0.0001f;
}

void kmeans_train(KMeansModel *model, Student *data, int count, NormalizeParams *norm) {
    kmeans_random_centroids(model, data, count, norm);
    
    for (int i = 0; i < 100; i++) {
        int converged = kmeans_step(model, data, count, norm);
        if (converged) {
            printf("KMeans converged at iteration %d\n", i + 1);
            break;
        }
    }
    
    model->sse = kmeans_compute_sse(model, data, count, norm);
    printf("KMeans SSE: %.4f\n", model->sse);
}

float kmeans_compute_sse(KMeansModel *model, Student *data, int count, NormalizeParams *norm) {
    float sse = 0;
    for (int i = 0; i < count; i++) {
        float features[FEATURE_DIM];
        normalize_student(&data[i], norm, features);
        int c = data[i].cluster;
        for (int d = 0; d < FEATURE_DIM; d++) {
            float diff = features[d] - model->centroids[c][d];
            sse += diff * diff;
        }
    }
    return sse;
}

int kmeans_predict(KMeansModel *model, const Student *s, NormalizeParams *norm) {
    float features[FEATURE_DIM];
    normalize_student(s, norm, features);
    return find_nearest_cluster(features, model);
}

// --- KNN ---

void knn_init(KNNModel *model, int k, Student *train_data, int train_count) {
    model->k = k;
    model->train_data = train_data;
    model->train_count = train_count;
}

float knn_distance(const float *a, const float *b, int dim) {
    float sum = 0;
    for (int i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

int knn_predict_cluster(KNNModel *model, const Student *s, NormalizeParams *norm) {
    float s_features[FEATURE_DIM];
    normalize_student(s, norm, s_features);
    
    float dists[2000];
    int labels[2000];
    int n = model->train_count;
    
    for (int i = 0; i < n; i++) {
        float t_features[FEATURE_DIM];
        normalize_student(&model->train_data[i], norm, t_features);
        dists[i] = knn_distance(s_features, t_features, FEATURE_DIM);
        labels[i] = model->train_data[i].cluster;
    }
    
    for (int i = 0; i < model->k; i++) {
        int min_idx = i;
        for (int j = i + 1; j < n; j++) {
            if (dists[j] < dists[min_idx]) min_idx = j;
        }
        float td = dists[i]; dists[i] = dists[min_idx]; dists[min_idx] = td;
        int tl = labels[i]; labels[i] = labels[min_idx]; labels[min_idx] = tl;
    }
    
    int votes[10] = {0};
    for (int i = 0; i < model->k; i++) {
        if (labels[i] >= 0 && labels[i] < 10) votes[labels[i]]++;
    }
    
    int best = 0, max_votes = 0;
    for (int i = 0; i < 10; i++) {
        if (votes[i] > max_votes) {
            max_votes = votes[i];
            best = i;
        }
    }
    return best;
}

float knn_predict_gpa(KNNModel *model, const Student *s, NormalizeParams *norm) {
    float s_features[FEATURE_DIM];
    normalize_student(s, norm, s_features);
    
    float dists[2000];
    float gpas[2000];
    int n = model->train_count;
    
    for (int i = 0; i < n; i++) {
        float t_features[FEATURE_DIM];
        normalize_student(&model->train_data[i], norm, t_features);
        dists[i] = knn_distance(s_features, t_features, FEATURE_DIM);
        gpas[i] = model->train_data[i].gpa_end;
    }
    
    for (int i = 0; i < model->k; i++) {
        int min_idx = i;
        for (int j = i + 1; j < n; j++) {
            if (dists[j] < dists[min_idx]) min_idx = j;
        }
        float td = dists[i]; dists[i] = dists[min_idx]; dists[min_idx] = td;
        float tg = gpas[i]; gpas[i] = gpas[min_idx]; gpas[min_idx] = tg;
    }
    
    float weighted_sum = 0, weight_total = 0;
    for (int i = 0; i < model->k; i++) {
        float w = 1.0f / (dists[i] + 0.001f);
        weighted_sum += w * gpas[i];
        weight_total += w;
    }
    return weighted_sum / weight_total;
}

float knn_evaluate_accuracy(KNNModel *model, Student *test_data, int test_count, NormalizeParams *norm) {
    int correct = 0;
    for (int i = 0; i < test_count; i++) {
        int pred = knn_predict_cluster(model, &test_data[i], norm);
        if (pred == test_data[i].cluster) correct++;
    }
    return (float)correct / test_count;
}

// ============================================================
// GUI Module (All English)
// ============================================================

#define COLOR_BG (Color){20, 20, 30, 255}
#define COLOR_PANEL (Color){40, 40, 55, 255}
#define COLOR_ACCENT (Color){100, 160, 255, 255}
#define COLOR_TEXT (Color){220, 220, 230, 255}
#define COLOR_TEXT_DIM (Color){150, 150, 170, 255}

#define CLUSTER_COLOR_0 (Color){100, 220, 120, 255}
#define CLUSTER_COLOR_1 (Color){255, 100, 100, 255}
#define CLUSTER_COLOR_2 (Color){100, 180, 255, 255}

static Rectangle btn_lr, btn_km, btn_knn;
static Rectangle btn_train_lr, btn_train_km, btn_predict_knn, btn_export;
static Rectangle input_rects[4];
static char input_buf[4][32] = {"3", "3.2", "10", "2"};
static int active_input = -1;
static int frame_counter = 0;

// Dimension selector for scatter plot
#define DIM_COUNT 6
static const char *dim_names[DIM_COUNT] = {"Year", "InitGPA", "AI Hours", "AI Purp", "FinalGPA", "Memory"};
static int scatter_x_dim = 2;  // default: AI Hours
static int scatter_y_dim = 4;  // default: Final GPA
static Rectangle dim_x_btns[DIM_COUNT];
static Rectangle dim_y_btns[DIM_COUNT];

void gui_init(AppState *state, int width, int height) {
    state->screen_width = width;
    state->screen_height = height;
    state->current_mode = MODE_LINEAR_REGRESSION;
    state->kmeans_iteration = 0;
    state->kmeans_training = 0;
    
    state->input_year = 3;
    state->input_gpa = 3.2f;
    state->input_hour = 10;
    state->input_purpose = 2;
    state->knn_predicted_cluster = -1;
    state->knn_predicted_gpa = 0;
    
    btn_lr = (Rectangle){20, 60, 200, 36};
    btn_km = (Rectangle){20, 106, 200, 36};
    btn_knn = (Rectangle){20, 152, 200, 36};
    
    btn_train_lr = (Rectangle){20, 200, 200, 36};
    btn_train_km = (Rectangle){20, 200, 200, 36};
    btn_predict_knn = (Rectangle){20, 380, 200, 36};
    btn_export = (Rectangle){20, height - 60, 200, 36};
    
    for (int i = 0; i < 4; i++) {
        input_rects[i] = (Rectangle){110, 230 + i * 38, 110, 28};
    }
}

static void draw_button(Rectangle r, const char *text, int active) {
    Color bg = active ? COLOR_ACCENT : COLOR_PANEL;
    Color text_color = active ? WHITE : COLOR_TEXT;
    
    DrawRectangleRec(r, bg);
    DrawRectangleLinesEx(r, 1, active ? (Color){150, 200, 255, 255} : (Color){70, 70, 90, 255});
    
    int text_w = MeasureText(text, 14);
    DrawText(text, (int)(r.x + r.width / 2 - text_w / 2), (int)(r.y + r.height / 2 - 7), 14, text_color);
}

static void draw_left_panel(AppState *state) {
    DrawRectangle(0, 0, 240, state->screen_height, COLOR_PANEL);
    DrawLine(240, 0, 240, state->screen_height, (Color){60, 60, 80, 255});
    
    DrawText("AI Student Analysis", 20, 20, 18, COLOR_ACCENT);
    DrawText("3-in-1 ML Demo", 20, 42, 12, COLOR_TEXT_DIM);
    
    draw_button(btn_lr, "Linear Regression", state->current_mode == MODE_LINEAR_REGRESSION);
    draw_button(btn_km, "KMeans Clustering", state->current_mode == MODE_KMEANS);
    draw_button(btn_knn, "KNN Prediction", state->current_mode == MODE_KNN);
    
    if (state->current_mode == MODE_LINEAR_REGRESSION) {
        draw_button(btn_train_lr, "Train Model", 0);
        DrawText("Learning rate: 0.01", 20, 250, 12, COLOR_TEXT_DIM);
        DrawText("Iterations: 1000", 20, 270, 12, COLOR_TEXT_DIM);
        DrawText("Features: 4 dims", 20, 290, 12, COLOR_TEXT_DIM);
    } else if (state->current_mode == MODE_KMEANS) {
        draw_button(btn_train_km, "Run KMeans", 0);
        DrawText("Clusters K = 3", 20, 250, 12, COLOR_TEXT_DIM);
        char iter_str[64];
        sprintf(iter_str, "Iteration: %d", state->kmeans_iteration);
        DrawText(iter_str, 20, 270, 12, COLOR_TEXT_DIM);
        DrawText("Features: normalized", 20, 290, 12, COLOR_TEXT_DIM);
    } else if (state->current_mode == MODE_KNN) {
        DrawText("Input new student:", 20, 205, 13, COLOR_TEXT);
        
        const char *labels[4] = {"Year(1-5)", "Init GPA", "AI Hours", "Purpose(0-4)"};
        for (int i = 0; i < 4; i++) {
            DrawText(labels[i], 20, 237 + i * 38, 12, COLOR_TEXT_DIM);
            Color border = active_input == i ? COLOR_ACCENT : (Color){70, 70, 90, 255};
            DrawRectangleRec(input_rects[i], (Color){30, 30, 45, 255});
            DrawRectangleLinesEx(input_rects[i], 1, border);
            DrawText(input_buf[i], input_rects[i].x + 6, input_rects[i].y + 7, 14, COLOR_TEXT);
        }
        
        draw_button(btn_predict_knn, "Predict", 0);
        DrawText("K = 5 neighbors", 20, 430, 12, COLOR_TEXT_DIM);
    }
    
    draw_button(btn_export, "Export CSV", 0);
    
    char info[64];
    sprintf(info, "Train set: %d", state->dataset.train_count);
    DrawText(info, 20, state->screen_height - 100, 11, COLOR_TEXT_DIM);
    sprintf(info, "Test set: %d", state->dataset.test_count);
    DrawText(info, 20, state->screen_height - 85, 11, COLOR_TEXT_DIM);
}

static Color get_cluster_color(int cluster) {
    switch (cluster) {
        case 0: return CLUSTER_COLOR_0;
        case 1: return CLUSTER_COLOR_1;
        case 2: return CLUSTER_COLOR_2;
        default: return GRAY;
    }
}

// Get value and min/max range for a specific dimension
static float get_dim_value(const Student *s, int dim) {
    switch (dim) {
        case 0: return (float)s->year;
        case 1: return s->gpa_start;
        case 2: return s->ai_hour;
        case 3: return (float)s->ai_purpose;
        case 4: return s->gpa_end;
        case 5: return s->memory;
        default: return 0;
    }
}

static void get_dim_range(const NormalizeParams *norm, int dim, float *out_min, float *out_max) {
    switch (dim) {
        case 0: *out_min = norm->age_min; *out_max = norm->age_max; break;
        case 1: *out_min = norm->gpa_min; *out_max = norm->gpa_max; break;
        case 2: *out_min = norm->hour_min; *out_max = norm->hour_max; break;
        case 3: *out_min = norm->purpose_min; *out_max = norm->purpose_max; break;
        case 4: *out_min = norm->gpa_min; *out_max = norm->gpa_max; break;
        case 5: *out_min = norm->memory_min; *out_max = norm->memory_max; break;
        default: *out_min = 0; *out_max = 1;
    }
}

// Get centroid value for a specific dimension (centroids are normalized 4D: year, gpa, hour, purpose)
static float get_centroid_dim(const KMeansModel *model, int cluster_idx, int dim, const NormalizeParams *norm) {
    float dmin, dmax;
    get_dim_range(norm, dim, &dmin, &dmax);
    // Map dimension index to centroid feature index
    int feat_idx = -1;
    if (dim == 0) feat_idx = 0;      // year
    else if (dim == 1) feat_idx = 1; // init gpa
    else if (dim == 2) feat_idx = 2; // ai hours
    else if (dim == 3) feat_idx = 3; // ai purpose
    else if (dim == 4) feat_idx = 1; // final gpa - approximate with init gpa centroid
    else if (dim == 5) feat_idx = 1; // memory - approximate with gpa
    
    if (feat_idx >= 0) {
        return model->centroids[cluster_idx][feat_idx] * (dmax - dmin) + dmin;
    }
    return (dmin + dmax) / 2;
}

static void draw_scatter_plot(AppState *state, int x, int y, int w, int h) {
    DrawRectangle(x, y, w, h, COLOR_BG);
    DrawRectangleLines(x, y, w, h, (Color){60, 60, 80, 255});
    
    char title[128];
    sprintf(title, "Scatter: %s vs %s", dim_names[scatter_x_dim], dim_names[scatter_y_dim]);
    DrawText(title, x + 10, y + 8, 14, COLOR_TEXT);
    
    // Draw dimension selector buttons at top
    DrawText("X-axis:", x + 180, y + 10, 11, COLOR_TEXT_DIM);
    DrawText("Y-axis:", x + 180, y + 28, 11, COLOR_TEXT_DIM);
    int btn_w = 62, btn_h = 18;
    for (int i = 0; i < DIM_COUNT; i++) {
        dim_x_btns[i] = (Rectangle){x + 230 + i * (btn_w + 4), y + 8, btn_w, btn_h};
        dim_y_btns[i] = (Rectangle){x + 230 + i * (btn_w + 4), y + 26, btn_w, btn_h};
        draw_button(dim_x_btns[i], dim_names[i], scatter_x_dim == i);
        draw_button(dim_y_btns[i], dim_names[i], scatter_y_dim == i);
    }
    
    int plot_x = x + 50, plot_y = y + 55;
    int plot_w = w - 70, plot_h = h - 80;
    
    for (int i = 0; i <= 5; i++) {
        int gx = plot_x + (int)(plot_w * i / 5.0f);
        DrawLine(gx, plot_y, gx, plot_y + plot_h, (Color){50, 50, 65, 255});
        int gy = plot_y + (int)(plot_h * i / 5.0f);
        DrawLine(plot_x, gy, plot_x + plot_w, gy, (Color){50, 50, 65, 255});
    }
    
    DrawLine(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, COLOR_TEXT_DIM);
    DrawLine(plot_x, plot_y, plot_x, plot_y + plot_h, COLOR_TEXT_DIM);
    
    DrawText(dim_names[scatter_x_dim], plot_x + plot_w / 2 - 30, plot_y + plot_h + 8, 11, COLOR_TEXT_DIM);
    DrawText(dim_names[scatter_y_dim], plot_x - 50, plot_y - 2, 11, COLOR_TEXT_DIM);
    
    NormalizeParams *norm = &state->dataset.norm;
    Student *data = state->dataset.train;
    int count = state->dataset.train_count;
    
    float xmin, xmax, ymin, ymax;
    get_dim_range(norm, scatter_x_dim, &xmin, &xmax);
    get_dim_range(norm, scatter_y_dim, &ymin, &ymax);
    float xrange = xmax - xmin;
    float yrange = ymax - ymin;
    if (xrange < 0.001f) xrange = 1;
    if (yrange < 0.001f) yrange = 1;
    
    for (int i = 0; i < count; i++) {
        float xv = get_dim_value(&data[i], scatter_x_dim);
        float yv = get_dim_value(&data[i], scatter_y_dim);
        float px = plot_x + plot_w * (xv - xmin) / xrange;
        float py = plot_y + plot_h * (1 - (yv - ymin) / yrange);
        
        Color c = get_cluster_color(data[i].cluster);
        DrawCircle((int)px, (int)py, 2.5f, c);
    }
    
    if (state->current_mode == MODE_KMEANS || state->current_mode == MODE_KNN) {
        for (int c = 0; c < state->km_model.k; c++) {
            float cx_val = get_centroid_dim(&state->km_model, c, scatter_x_dim, norm);
            float cy_val = get_centroid_dim(&state->km_model, c, scatter_y_dim, norm);
            
            float cx = plot_x + plot_w * (cx_val - xmin) / xrange;
            float cy = plot_y + plot_h * (1 - (cy_val - ymin) / yrange);
            
            Color cc = get_cluster_color(c);
            DrawCircle((int)cx, (int)cy, 8, WHITE);
            DrawCircle((int)cx, (int)cy, 6, cc);
        }
    }
    
    int legend_x = x + w - 130, legend_y = y + 55;
    DrawRectangle(legend_x - 5, legend_y - 5, 125, 75, (Color){30, 30, 45, 200});
    
    DrawCircle(legend_x + 8, legend_y + 8, 5, CLUSTER_COLOR_0);
    DrawText("Efficient", legend_x + 20, legend_y + 3, 11, COLOR_TEXT);
    
    DrawCircle(legend_x + 8, legend_y + 28, 5, CLUSTER_COLOR_1);
    DrawText("AI-Dependent", legend_x + 20, legend_y + 23, 11, COLOR_TEXT);
    
    DrawCircle(legend_x + 8, legend_y + 48, 5, CLUSTER_COLOR_2);
    DrawText("Low AI Use", legend_x + 20, legend_y + 43, 11, COLOR_TEXT);
}

static void draw_loss_curve(AppState *state, int x, int y, int w, int h) {
    DrawRectangle(x, y, w, h, COLOR_BG);
    DrawRectangleLines(x, y, w, h, (Color){60, 60, 80, 255});
    
    DrawText("Training Loss Curve (MSE)", x + 10, y + 8, 14, COLOR_TEXT);
    
    int plot_x = x + 50, plot_y = y + 35;
    int plot_w = w - 70, plot_h = h - 60;
    
    for (int i = 0; i <= 5; i++) {
        int gx = plot_x + (int)(plot_w * i / 5.0f);
        DrawLine(gx, plot_y, gx, plot_y + plot_h, (Color){50, 50, 65, 255});
        int gy = plot_y + (int)(plot_h * i / 5.0f);
        DrawLine(plot_x, gy, plot_x + plot_w, gy, (Color){50, 50, 65, 255});
    }
    
    DrawLine(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, COLOR_TEXT_DIM);
    DrawLine(plot_x, plot_y, plot_x, plot_y + plot_h, COLOR_TEXT_DIM);
    
    DrawText("Iterations", plot_x + plot_w / 2 - 30, plot_y + plot_h + 8, 11, COLOR_TEXT_DIM);
    DrawText("MSE", plot_x - 30, plot_y - 2, 11, COLOR_TEXT_DIM);
    
    LinearModel *m = &state->lr_model;
    if (m->loss_count < 2) return;
    
    float max_loss = m->loss_history[0];
    float min_loss = m->loss_history[m->loss_count - 1];
    float range = max_loss - min_loss;
    if (range < 0.001f) range = 0.001f;
    
    for (int i = 0; i < m->loss_count - 1; i++) {
        float x1 = plot_x + (float)plot_w * i / (m->loss_count - 1);
        float y1 = plot_y + plot_h - (float)plot_h * (m->loss_history[i] - min_loss) / range;
        float x2 = plot_x + (float)plot_w * (i + 1) / (m->loss_count - 1);
        float y2 = plot_y + plot_h - (float)plot_h * (m->loss_history[i + 1] - min_loss) / range;
        DrawLine((int)x1, (int)y1, (int)x2, (int)y2, COLOR_ACCENT);
    }
}

static void draw_fit_scatter(AppState *state, int x, int y, int w, int h) {
    DrawRectangle(x, y, w, h, COLOR_BG);
    DrawRectangleLines(x, y, w, h, (Color){60, 60, 80, 255});
    
    DrawText("Actual vs Predicted GPA", x + 10, y + 8, 14, COLOR_TEXT);
    
    int plot_x = x + 50, plot_y = y + 35;
    int plot_w = w - 70, plot_h = h - 60;
    
    DrawLine(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y, (Color){100, 100, 120, 255});
    
    DrawLine(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, COLOR_TEXT_DIM);
    DrawLine(plot_x, plot_y, plot_x, plot_y + plot_h, COLOR_TEXT_DIM);
    
    DrawText("Actual GPA", plot_x + plot_w / 2 - 35, plot_y + plot_h + 8, 11, COLOR_TEXT_DIM);
    DrawText("Predicted", plot_x - 50, plot_y - 2, 11, COLOR_TEXT_DIM);
    
    Student *data = state->dataset.test;
    int count = state->dataset.test_count;
    NormalizeParams *norm = &state->dataset.norm;
    
    float gpa_min = norm->gpa_min;
    float gpa_max = norm->gpa_max;
    float range = gpa_max - gpa_min;
    if (range < 0.1f) range = 0.1f;
    
    for (int i = 0; i < count; i++) {
        float actual = data[i].gpa_end;
        float pred = linear_predict(&state->lr_model, &data[i], norm);
        
        float px = plot_x + plot_w * (actual - gpa_min) / range;
        float py = plot_y + plot_h - plot_h * (pred - gpa_min) / range;
        
        DrawCircle((int)px, (int)py, 2.0f, COLOR_ACCENT);
    }
}

static void draw_bottom_right_lr(AppState *state, int x, int y, int w, int h) {
    DrawRectangle(x, y, w, h, COLOR_PANEL);
    DrawRectangleLines(x, y, w, h, (Color){60, 60, 80, 255});
    
    DrawText("Regression Analysis", x + 12, y + 10, 15, COLOR_ACCENT);
    
    LinearModel *m = &state->lr_model;
    int py = y + 40;
    
    DrawText("Feature Weights:", x + 12, py, 13, COLOR_TEXT);
    py += 20;
    
    char buf[128];
    sprintf(buf, "Year:        %+.4f", m->w_age);
    DrawText(buf, x + 20, py, 12, COLOR_TEXT); py += 18;
    
    sprintf(buf, "Init GPA:    %+.4f", m->w_gpa);
    DrawText(buf, x + 20, py, 12, COLOR_TEXT); py += 18;
    
    Color hour_color = m->w_hour < 0 ? (Color){255, 120, 120, 255} : COLOR_TEXT;
    sprintf(buf, "AI Hours:    %+.4f", m->w_hour);
    DrawText(buf, x + 20, py, 12, hour_color); py += 18;
    
    Color purpose_color = m->w_purpose < 0 ? (Color){255, 120, 120, 255} : COLOR_TEXT;
    sprintf(buf, "AI Purpose:  %+.4f", m->w_purpose);
    DrawText(buf, x + 20, py, 12, purpose_color); py += 18;
    
    sprintf(buf, "Bias:        %+.4f", m->bias);
    DrawText(buf, x + 20, py, 12, COLOR_TEXT); py += 22;
    
    float test_mse = linear_evaluate(m, state->dataset.test, state->dataset.test_count, &state->dataset.norm);
    sprintf(buf, "Test MSE: %.4f", test_mse);
    DrawText(buf, x + 12, py, 13, COLOR_ACCENT); py += 25;
    
    DrawText("Conclusion:", x + 12, py, 13, COLOR_TEXT); py += 18;
    if (m->w_hour < 0) {
        DrawText("More AI hours -> lower GPA", x + 20, py, 11, (Color){255, 150, 150, 255});
        py += 16;
    }
    if (m->w_purpose < 0) {
        DrawText("Worse AI use -> lower score", x + 20, py, 11, (Color){255, 150, 150, 255});
        py += 16;
    }
    DrawText("Init GPA is strongest factor", x + 20, py, 11, COLOR_TEXT_DIM);
}

static void draw_kmeans_info(AppState *state, int x, int y, int w, int h) {
    DrawRectangle(x, y, w, h, COLOR_PANEL);
    DrawRectangleLines(x, y, w, h, (Color){60, 60, 80, 255});
    
    DrawText("KMeans Results", x + 12, y + 10, 15, COLOR_ACCENT);
    
    KMeansModel *m = &state->km_model;
    int py = y + 40;
    
    const char *cluster_names[3] = {"Efficient Learners", "AI-Dependent", "Low AI Users"};
    Color cluster_colors[3] = {CLUSTER_COLOR_0, CLUSTER_COLOR_1, CLUSTER_COLOR_2};
    
    for (int i = 0; i < m->k && i < 3; i++) {
        DrawCircle(x + 22, py + 6, 5, cluster_colors[i]);
        char buf[64];
        sprintf(buf, "%s: %d", cluster_names[i], m->cluster_counts[i]);
        DrawText(buf, x + 35, py, 12, COLOR_TEXT);
        py += 22;
    }
    
    py += 5;
    char sse_buf[64];
    sprintf(sse_buf, "SSE: %.2f", m->sse);
    DrawText(sse_buf, x + 12, py, 13, COLOR_ACCENT);
    py += 25;
    
    DrawText("Cluster Profiles:", x + 12, py, 13, COLOR_TEXT); py += 18;
    DrawText("Green: balanced AI, high GPA", x + 20, py, 11, COLOR_TEXT_DIM); py += 15;
    DrawText("Red: heavy AI use, low score", x + 20, py, 11, COLOR_TEXT_DIM); py += 15;
    DrawText("Blue: little AI, mid GPA", x + 20, py, 11, COLOR_TEXT_DIM);
}

static void draw_knn_result(AppState *state, int x, int y, int w, int h) {
    DrawRectangle(x, y, w, h, COLOR_PANEL);
    DrawRectangleLines(x, y, w, h, (Color){60, 60, 80, 255});
    
    DrawText("KNN Prediction", x + 12, y + 10, 15, COLOR_ACCENT);
    
    int py = y + 40;
    
    if (state->knn_predicted_cluster < 0) {
        DrawText("Enter data and click Predict", x + 12, py, 12, COLOR_TEXT_DIM);
        return;
    }
    
    const char *cluster_names[3] = {"Efficient Learner", "AI-Dependent", "Low AI User"};
    Color cluster_colors[3] = {CLUSTER_COLOR_0, CLUSTER_COLOR_1, CLUSTER_COLOR_2};
    int c = state->knn_predicted_cluster;
    
    DrawText("Category:", x + 12, py, 13, COLOR_TEXT);
    DrawCircle(x + 90, py + 7, 6, cluster_colors[c]);
    DrawText(cluster_names[c], x + 105, py, 13, cluster_colors[c]);
    py += 28;
    
    char buf[64];
    sprintf(buf, "Predicted GPA: %.3f", state->knn_predicted_gpa);
    DrawText(buf, x + 12, py, 13, COLOR_ACCENT);
    py += 28;
    
    DrawText("AI Usage Advice:", x + 12, py, 13, COLOR_TEXT); py += 18;
    
    if (c == 0) {
        DrawText("Keep up the good work!", x + 20, py, 11, COLOR_TEXT_DIM); py += 15;
        DrawText("Balanced AI learning", x + 20, py, 11, CLUSTER_COLOR_0);
        py += 20;
        DrawText("Risk Level: LOW", x + 12, py, 12, CLUSTER_COLOR_0);
    } else if (c == 1) {
        DrawText("Reduce AI dependency", x + 20, py, 11, COLOR_TEXT_DIM); py += 15;
        DrawText("Use AI only for reference", x + 20, py, 11, CLUSTER_COLOR_1);
        py += 20;
        DrawText("Risk Level: HIGH", x + 12, py, 12, CLUSTER_COLOR_1);
    } else {
        DrawText("Try AI to boost efficiency", x + 20, py, 11, COLOR_TEXT_DIM); py += 15;
        DrawText("Moderate AI use helps", x + 20, py, 11, CLUSTER_COLOR_2);
        py += 20;
        DrawText("Risk Level: MEDIUM", x + 12, py, 12, CLUSTER_COLOR_2);
    }
}

void gui_render(AppState *state) {
    BeginDrawing();
    ClearBackground(COLOR_BG);
    
    draw_left_panel(state);
    
    int right_x = 250;
    int right_w = state->screen_width - 260;
    int top_h = (state->screen_height - 40) / 2;
    
    if (state->current_mode == MODE_LINEAR_REGRESSION) {
        draw_loss_curve(state, right_x, 10, right_w, top_h);
        draw_bottom_right_lr(state, right_x, top_h + 20, right_w / 2 - 5, top_h - 10);
        draw_fit_scatter(state, right_x + right_w / 2 + 5, top_h + 20, right_w / 2 - 5, top_h - 10);
    } else if (state->current_mode == MODE_KMEANS) {
        draw_scatter_plot(state, right_x, 10, right_w, top_h);
        draw_kmeans_info(state, right_x, top_h + 20, right_w / 2, top_h - 10);
    } else if (state->current_mode == MODE_KNN) {
        draw_scatter_plot(state, right_x, 10, right_w, top_h);
        draw_knn_result(state, right_x, top_h + 20, right_w / 2, top_h - 10);
    }
    
    EndDrawing();
}

void gui_handle_input(AppState *state) {
    frame_counter++;
    
    Vector2 mouse = GetMousePosition();
    int clicked = IsMouseButtonPressed(MOUSE_LEFT_BUTTON);
    
    if (clicked && CheckCollisionPointRec(mouse, btn_lr)) state->current_mode = MODE_LINEAR_REGRESSION;
    if (clicked && CheckCollisionPointRec(mouse, btn_km)) state->current_mode = MODE_KMEANS;
    if (clicked && CheckCollisionPointRec(mouse, btn_knn)) state->current_mode = MODE_KNN;
    
    // Dimension selector clicks (only in KMeans/KNN modes where scatter is shown)
    if (state->current_mode == MODE_KMEANS || state->current_mode == MODE_KNN) {
        if (clicked) {
            for (int i = 0; i < DIM_COUNT; i++) {
                if (CheckCollisionPointRec(mouse, dim_x_btns[i])) {
                    scatter_x_dim = i;
                    break;
                }
                if (CheckCollisionPointRec(mouse, dim_y_btns[i])) {
                    scatter_y_dim = i;
                    break;
                }
            }
        }
    }
    
    if (state->current_mode == MODE_LINEAR_REGRESSION && clicked && CheckCollisionPointRec(mouse, btn_train_lr)) {
        linear_train(&state->lr_model, state->dataset.train, state->dataset.train_count, &state->dataset.norm);
    }
    
    if (state->current_mode == MODE_KMEANS && clicked && CheckCollisionPointRec(mouse, btn_train_km)) {
        kmeans_random_centroids(&state->km_model, state->dataset.train, state->dataset.train_count, &state->dataset.norm);
        state->kmeans_iteration = 0;
        state->kmeans_training = 1;
    }
    
    if (state->current_mode == MODE_KMEANS && state->kmeans_training && frame_counter % 10 == 0) {
        if (state->kmeans_iteration < 50) {
            int converged = kmeans_step(&state->km_model, state->dataset.train, state->dataset.train_count, &state->dataset.norm);
            state->kmeans_iteration++;
            if (converged) {
                state->kmeans_training = 0;
                state->km_model.sse = kmeans_compute_sse(&state->km_model, state->dataset.train, state->dataset.train_count, &state->dataset.norm);
            }
        } else {
            state->kmeans_training = 0;
            state->km_model.sse = kmeans_compute_sse(&state->km_model, state->dataset.train, state->dataset.train_count, &state->dataset.norm);
        }
    }
    
    if (state->current_mode == MODE_KNN) {
        if (clicked) {
            active_input = -1;
            for (int i = 0; i < 4; i++) {
                if (CheckCollisionPointRec(mouse, input_rects[i])) {
                    active_input = i;
                    break;
                }
            }
        }
        
        if (active_input >= 0) {
            int key = GetCharPressed();
            while (key > 0) {
                int len = (int)strlen(input_buf[active_input]);
                if ((key >= '0' && key <= '9') || key == '.') {
                    if (len < 30) {
                        input_buf[active_input][len] = (char)key;
                        input_buf[active_input][len + 1] = '\0';
                    }
                }
                key = GetCharPressed();
            }
            if (IsKeyPressed(KEY_BACKSPACE)) {
                int len = (int)strlen(input_buf[active_input]);
                if (len > 0) input_buf[active_input][len - 1] = '\0';
            }
        }
        
        if (clicked && CheckCollisionPointRec(mouse, btn_predict_knn)) {
            state->input_year = atoi(input_buf[0]);
            state->input_gpa = atof(input_buf[1]);
            state->input_hour = atof(input_buf[2]);
            state->input_purpose = atoi(input_buf[3]);
            
            Student s = {0};
            s.year = state->input_year;
            s.gpa_start = state->input_gpa;
            s.ai_hour = state->input_hour;
            s.ai_purpose = state->input_purpose;
            
            state->knn_predicted_cluster = knn_predict_cluster(&state->knn_model, &s, &state->dataset.norm);
            state->knn_predicted_gpa = knn_predict_gpa(&state->knn_model, &s, &state->dataset.norm);
        }
    }
    
    if (clicked && CheckCollisionPointRec(mouse, btn_export)) {
        if (state->current_mode == MODE_KMEANS) {
            export_clusters_csv("cluster_results.csv", state->dataset.train, state->dataset.train_count);
        } else if (state->current_mode == MODE_LINEAR_REGRESSION) {
            int n = state->dataset.test_count;
            float *preds = (float *)malloc(sizeof(float) * n);
            for (int i = 0; i < n; i++) {
                preds[i] = linear_predict(&state->lr_model, &state->dataset.test[i], &state->dataset.norm);
            }
            export_predictions_csv("prediction_results.csv", state->dataset.test, n, preds);
            free(preds);
        }
    }
}

void gui_cleanup(AppState *state) {
    (void)state;
}

// ============================================================
// Main
// ============================================================

#define CSV_FILE "student_data.csv"
#define SCREEN_WIDTH 1400
#define SCREEN_HEIGHT 900

static void init_models(AppState *state) {
    linear_model_init(&state->lr_model, 0.01f, 1000);
    kmeans_init(&state->km_model, 3);
    knn_init(&state->knn_model, 5, state->dataset.train, state->dataset.train_count);
}

static void pretrain_models(AppState *state) {
    printf("\n=== Pre-training models ===\n");
    
    linear_train(&state->lr_model, state->dataset.train, state->dataset.train_count, &state->dataset.norm);
    kmeans_train(&state->km_model, state->dataset.train, state->dataset.train_count, &state->dataset.norm);
    
    for (int i = 0; i < state->dataset.test_count; i++) {
        state->dataset.test[i].cluster = kmeans_predict(&state->km_model, &state->dataset.test[i], &state->dataset.norm);
    }
    
    float acc = knn_evaluate_accuracy(&state->knn_model, state->dataset.test, state->dataset.test_count, &state->dataset.norm);
    printf("KNN Accuracy: %.2f%%\n", acc * 100);
    
    printf("=== Pre-training complete ===\n\n");
}

int main(void) {
    printf("AI Student Performance Analysis System\n");
    printf("=======================================\n");
    
    Student all_students[MAX_STUDENTS];
    int total = load_csv(CSV_FILE, all_students, MAX_STUDENTS);
    if (total <= 0) {
        printf("Error: Cannot load %s\n", CSV_FILE);
        printf("Make sure student_data.csv is in the same folder.\n");
        return 1;
    }
    
    AppState state;
    compute_normalize_params(all_students, total, &state.dataset.norm);
    
    split_dataset(all_students, total,
                  &state.dataset.train, &state.dataset.train_count,
                  &state.dataset.test, &state.dataset.test_count,
                  0.8f);
    
    init_models(&state);
    pretrain_models(&state);
    
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "AI Student Analysis - Linear Regression + KMeans + KNN");
    SetTargetFPS(60);
    
    gui_init(&state, SCREEN_WIDTH, SCREEN_HEIGHT);
    
    while (!WindowShouldClose()) {
        gui_handle_input(&state);
        gui_render(&state);
    }
    
    CloseWindow();
    linear_model_free(&state.lr_model);
    free(state.dataset.train);
    free(state.dataset.test);
    
    printf("Program exited.\n");
    return 0;
}
