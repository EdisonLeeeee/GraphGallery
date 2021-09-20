#include <torch/extension.h>
#define CHECK_CPU(x) AT_ASSERTM(x.device().is_cpu(), #x " must be CPU tensor")
#define CHECK_INPUT(x) AT_ASSERTM(x, "Input mismatch")

#define AT_DISPATCH_HAS_VALUE(optional_value, ...) \
    [&] {                                          \
        if (optional_value.has_value())            \
        {                                          \
            const bool HAS_VALUE = true;           \
            return __VA_ARGS__();                  \
        }                                          \
        else                                       \
        {                                          \
            const bool HAS_VALUE = false;          \
            return __VA_ARGS__();                  \
        }                                          \
    }()

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> neighbor_sampler_cpu(torch::Tensor rowptr, torch::Tensor col, torch::Tensor idx,
                                                                             int64_t num_neighbors, bool replace);

// Returns `targets`, `neighbors`, `e_id`
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> neighbor_sampler_cpu(torch::Tensor rowptr, torch::Tensor col, torch::Tensor idx,
                                                                             int64_t num_neighbors, bool replace)
{
    CHECK_CPU(rowptr);
    CHECK_CPU(col);
    CHECK_CPU(idx);
    CHECK_INPUT(idx.dim() == 1);

    auto rowptr_data = rowptr.data_ptr<int64_t>();
    auto col_data = col.data_ptr<int64_t>();
    auto idx_data = idx.data_ptr<int64_t>();

    std::vector<int64_t> neighbors;
    std::vector<int64_t> targets;
    std::vector<int64_t> e_id;

    int64_t i;

    int64_t n, c, e, row_start, row_end, row_count;

    if (num_neighbors < 0)
    { // No sampling ======================================

        for (int64_t i = 0; i < idx.numel(); i++)
        {
            n = idx_data[i];
            row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
            row_count = row_end - row_start;

            for (int64_t j = 0; j < row_count; j++)
            {
                e = row_start + j;
                c = col_data[e];
                targets.push_back(n);
                neighbors.push_back(c);
                e_id.push_back(e);
            }
        }
    }

    else if (replace)
    { // Sample with replacement ===============================
        for (int64_t i = 0; i < idx.numel(); i++)
        {
            n = idx_data[i];
            row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
            row_count = row_end - row_start;

            std::unordered_set<int64_t> perm;
            if (row_count <= num_neighbors)
            {
                for (int64_t j = 0; j < row_count; j++)
                    perm.insert(j);
                for (int64_t j = 0; j < num_neighbors - row_count; j++)
                {
                    e = row_start + rand() % row_count;
                    c = col_data[e];
                    targets.push_back(n);
                    neighbors.push_back(c);
                    e_id.push_back(e);
                }
            }
            else
            { // See: https://www.nowherenearithaca.com/2013/05/
                //      robert-floyds-tiny-and-beautiful.html
                for (int64_t j = row_count - num_neighbors; j < row_count; j++)
                {
                    if (!perm.insert(rand() % j).second)
                        perm.insert(j);
                }
            }

            for (const int64_t &p : perm)
            {
                e = row_start + p;
                c = col_data[e];
                targets.push_back(n);
                neighbors.push_back(c);
                e_id.push_back(e);
            }
        }
    }
    else
    { // Sample without replacement via Robert Floyd algorithm ============

        for (int64_t i = 0; i < idx.numel(); i++)
        {
            n = idx_data[i];
            row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
            row_count = row_end - row_start;

            std::unordered_set<int64_t> perm;
            if (row_count <= num_neighbors)
            {
                for (int64_t j = 0; j < row_count; j++)
                    perm.insert(j);
            }
            else
            { // See: https://www.nowherenearithaca.com/2013/05/
                //      robert-floyds-tiny-and-beautiful.html
                for (int64_t j = row_count - num_neighbors; j < row_count; j++)
                {
                    if (!perm.insert(rand() % j).second)
                        perm.insert(j);
                }
            }

            for (const int64_t &p : perm)
            {
                e = row_start + p;
                c = col_data[e];
                targets.push_back(n);
                neighbors.push_back(c);
                e_id.push_back(e);
            }
        }
    }

    int64_t N = neighbors.size();
    auto out_target_id = torch::from_blob(targets.data(), {N}, col.options()).clone();
    auto out_neighbor_id = torch::from_blob(neighbors.data(), {N}, col.options()).clone();
    auto out_e_id = torch::from_blob(e_id.data(), {N}, col.options()).clone();

    return std::make_tuple(out_target_id, out_neighbor_id, out_e_id);
}

PYBIND11_MODULE(sampler, m)
{
    m.def("neighbor_sampler_cpu", &neighbor_sampler_cpu, "Neighborhood Sampler");
}