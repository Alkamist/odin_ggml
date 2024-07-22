package main

import "base:runtime"
import "core:c"
import "core:os"
import "core:fmt"
import "core:mem"
import "core:math"
import "core:math/rand"
import "core:slice"
import "core:strconv"
import "core:encoding/csv"
// import "core:image"
// import "core:image/png"
import "ggml"

INPUT_SIZE :: 784
OUTPUT_SIZE :: 10

BATCH_SIZE :: 16
THREAD_COUNT :: 1

Layer :: struct {
    weights: ^ggml.tensor,
    biases: ^ggml.tensor,
}

main :: proc() {
    defer fmt.println("Done")
    defer free_all(context.temp_allocator)

    backend := ggml.backend_cpu_init()

    if ggml.backend_is_cpu(backend) {
        ggml.backend_cpu_set_n_threads(backend, THREAD_COUNT)
    }

    dimensions := [?][2]int{
        {INPUT_SIZE, 500},
        {500, OUTPUT_SIZE},
    }

    layers := make([]Layer, len(dimensions))
    defer delete(layers)

    build_graph :: proc(layers: []Layer, ctx: ^ggml.ggml_context, input: ^ggml.tensor) -> ^ggml.tensor {
        layer_input := input
        for layer in layers {
            t := ggml.mul_mat(ctx, layer.weights, layer_input)
            t = ggml.add(ctx, t, ggml.repeat(ctx, layer.biases, t))
            layer_input = ggml.relu(ctx, t)
        }
        return ggml.soft_max(ctx, layer_input)
    }

    layer_ctx := ggml.init({
        mem_size = ggml.tensor_overhead() * len(layers) * 4,
        mem_buffer = nil,
        no_alloc = true,
    })
    defer ggml.free(layer_ctx)

    for dimension, i in dimensions {
        layers[i].weights = ggml.new_tensor_2d(layer_ctx, .F32, i64(dimension[0]), i64(dimension[1]))
        layers[i].biases = ggml.new_tensor_1d(layer_ctx, .F32, i64(dimension[1]))
        ggml.set_param(layer_ctx, layers[i].weights)
        ggml.set_param(layer_ctx, layers[i].biases)
    }

    layer_data := ggml.backend_alloc_ctx_tensors(layer_ctx, backend)

    // load_model(layers, "ggml-model-f32.bin")

    { // Training
        Training_State :: struct {
            data_set: Data_Set,
            optimizer: ggml.opt_context,
            input: ^ggml.tensor,
            target: ^ggml.tensor,
        }

        training_state: Training_State

        for layer in layers {
            tensor_randomize_normal(layer.weights, 0, 1)
            tensor_randomize_normal(layer.biases, 0, 1)
        }

        input_ctx := ggml.init({
            mem_size = ggml.tensor_overhead() * 2,
            mem_buffer = nil,
            no_alloc = true,
        })
        defer ggml.free(input_ctx)

        training_state.input = ggml.new_tensor_2d(input_ctx, .F32, INPUT_SIZE, BATCH_SIZE)
        ggml.set_input(training_state.input)

        training_state.target = ggml.new_tensor_2d(input_ctx, .F32, OUTPUT_SIZE, BATCH_SIZE)
        ggml.backend_alloc_ctx_tensors(input_ctx, backend)

        compute_ctx := ggml.init({
            mem_size = 1024 * 1024,
            mem_buffer = nil,
            no_alloc = true,
        })
        defer ggml.free(compute_ctx)

        probs := build_graph(layers, compute_ctx, training_state.input)
        loss := ggml.cross_entropy_loss(compute_ctx, probs, training_state.target)

        gf := ggml.new_graph_custom(compute_ctx, ggml.DEFAULT_GRAPH_SIZE, true)
        ggml.build_forward_expand(gf, loss)

        gb := ggml.graph_dup(compute_ctx, gf)
        ggml.build_backward_expand(compute_ctx, gf, gb, true)

        gallocr := ggml.gallocr_new(ggml.backend_get_default_buffer_type(backend))
        defer ggml.gallocr_free(gallocr)

        ggml.gallocr_reserve(gallocr, gb)
        ggml.gallocr_alloc_graph(gallocr, gb)

        data_set_init(&training_state.data_set, 60000, context.temp_allocator)
        if !data_set_load_csv(&training_state.data_set, "mnist_train.csv") {
            fmt.eprintln("Failed to load mnist_train.csv")
        }

        opt_params := ggml.opt_default_params(.ADAM)
        opt_params.print_forward_graph = false
        opt_params.print_backward_graph = false
        opt_params.adam.n_iter = 60000 * 5

        parameter_count := 0
        for layer in layers {
            parameter_count += int(layer.weights.ne[0] * layer.weights.ne[1])
            parameter_count += int(layer.biases.ne[0])
        }

        ggml.opt_init(training_state.optimizer.ctx, &training_state.optimizer, opt_params, i64(parameter_count))

        work_ctx := ggml.init({
            mem_size = ggml.graph_plan(gb, THREAD_COUNT).work_size + ggml.OBJECT_SIZE,
            mem_buffer = nil,
            no_alloc = false,
        })
        defer ggml.free(work_ctx)

        ggml.opt_resume_g(work_ctx, &training_state.optimizer, loss, gf, gb, proc "c" (data: rawptr, accum_step: c.int, sched: ^f32, cancel: ^bool) {
            context = runtime.default_context()
            state := cast(^Training_State)data

            index := rand.int_max(state.data_set.size - BATCH_SIZE)

            tensor_set(state.input, state.data_set.input[index * INPUT_SIZE:][:INPUT_SIZE * BATCH_SIZE])
            tensor_set(state.target, state.data_set.target[index * OUTPUT_SIZE:][:OUTPUT_SIZE * BATCH_SIZE])
        }, &training_state)
    }

    { // Testing
        test_ctx := ggml.init({
            mem_size = 1024 * 1024,
            mem_buffer = nil,
            no_alloc = true,
        })
        defer ggml.free(test_ctx)

        input := ggml.new_tensor_1d(test_ctx, .F32, INPUT_SIZE)
        probs := build_graph(layers, test_ctx, input)

        gf := ggml.new_graph(test_ctx)
        ggml.build_forward_expand(gf, probs)

        gallocr := ggml.gallocr_new(ggml.backend_get_default_buffer_type(backend))
        defer ggml.gallocr_free(gallocr)

        ggml.gallocr_reserve(gallocr, gf)
        ggml.gallocr_alloc_graph(gallocr, gf)

        test_set: Data_Set
        data_set_init(&test_set, 10000, context.temp_allocator)
        if !data_set_load_csv(&test_set, "mnist_test.csv") {
            fmt.eprintln("Failed to load mnist_test.csv")
        }

        correct_answer_count := 0
        for i in 0 ..< test_set.size {
            tensor_set(input, test_set.input[i * INPUT_SIZE:][:INPUT_SIZE])

            ggml.backend_graph_compute(backend, gf)

            prediction: [OUTPUT_SIZE]f32
            tensor_get(probs, prediction[:])

            if slice.max_index(prediction[:]) == slice.max_index(test_set.target[i * OUTPUT_SIZE:][:OUTPUT_SIZE]) {
                correct_answer_count += 1
            }
        }

        fmt.printfln("Accuracy is %v%%", 100.0 * f32(correct_answer_count) / f32(test_set.size))
    }
}

Data_Set :: struct {
    size: int,
    input: []f32,
    target: []f32,
}

data_set_init :: proc(data_set: ^Data_Set, size: int, allocator := context.allocator) {
    data_set.size = size
    data_set.input = make([]f32, size * INPUT_SIZE, allocator)
    data_set.target = make([]f32, size * OUTPUT_SIZE, allocator)
}

data_set_destroy :: proc(data_set: ^Data_Set) {
    delete(data_set.input)
    delete(data_set.target)
}

data_set_load_csv :: proc(data_set: ^Data_Set, file_name: string) -> (ok: bool) {
    file_data, success := os.read_entire_file(file_name)
    if !success {
        return
    }

    csv_reader: csv.Reader
    csv.reader_init_with_string(&csv_reader, cast(string)file_data, context.temp_allocator)
    defer csv.reader_destroy(&csv_reader)

    _, _ = csv.read(&csv_reader)

    for i in 0 ..< data_set.size {
        values_str, err := csv.read(&csv_reader)
        if err != nil {
            break
        }

        y_int, _ := strconv.parse_i64(values_str[0])
        data_set.target[i * OUTPUT_SIZE + int(y_int)] = 1

        for j in 0 ..< INPUT_SIZE {
            value_int, _ := strconv.parse_i64(values_str[j + 1])
            data_set.input[i * INPUT_SIZE + j] = f32(value_int) / 255.0
        }
    }

    ok = true
    return
}

tensor_get :: proc(tensor: ^ggml.tensor, data: []f32) {
    assert(i64(len(data)) == ggml.nelements(tensor))
    for i1 in 0 ..< tensor.ne[1] {
        for i0 in 0 ..< tensor.ne[0] {
            data[i1 * tensor.ne[0] + i0] = ggml.get_f32_nd(tensor, i32(i0), i32(i1), 0, 0)
        }
    }
}

tensor_set :: proc(tensor: ^ggml.tensor, data: []f32) {
    assert(i64(len(data)) == ggml.nelements(tensor))
    for i1 in 0 ..< tensor.ne[1] {
        for i0 in 0 ..< tensor.ne[0] {
            ggml.set_f32_nd(tensor, i32(i0), i32(i1), 0, 0, data[i1 * tensor.ne[0] + i0])
        }
    }
}

tensor_print :: proc(tensor: ^ggml.tensor) {
    data := make([]f32, ggml.nelements(tensor))
    defer delete(data)
    for i1 in 0 ..< tensor.ne[1] {
        for i0 in 0 ..< tensor.ne[0] {
            data[i1 * tensor.ne[0] + i0] = ggml.get_f32_nd(tensor, i32(i0), i32(i1), 0, 0)
        }
    }
    fmt.println(data)
}

tensor_randomize_normal :: proc(tensor: ^ggml.tensor, mean, stddev: f32) -> ^ggml.tensor {
    element_count := ggml.nelements(tensor)
    data := make([]f32, element_count, context.temp_allocator)

    // Xavier
    scale := 1.0 / math.sqrt_f32(f32(element_count))
    for i in 0 ..< len(data) {
        data[i] = scale * rand.float32_normal(mean, stddev)
    }

    tensor_set(tensor, data)

    return tensor
}

load_model :: proc(layers: []Layer, file_name: string) -> bool {
    file_handle, file_err := os.open(file_name)
    if file_err != os.ERROR_NONE {
        fmt.eprintfln("Failed to open %v", file_name)
        return false
    }
    defer os.close(file_handle)

    // Verify magic.
    magic: u32
    os.read_ptr(file_handle, &magic, size_of(u32))
    if magic != ggml.FILE_MAGIC {
        fmt.eprintfln("Invalid model file %v (bad magic)", file_name)
        return false
    }

    // FC1
    {
        dimension_count: i32
        os.read_ptr(file_handle, &dimension_count, size_of(i32))

        ne_weight := [2]i32{1, 1}
        for i in 0 ..< dimension_count {
            os.read_ptr(file_handle, &ne_weight[i], size_of(i32))
        }
        os.read_ptr(file_handle, layers[0].weights.data, int(ggml.nbytes(layers[0].weights)))

        ne_bias := [2]i32{1, 1}
        for i in 0 ..< dimension_count {
            os.read_ptr(file_handle, &ne_bias[i], size_of(i32))
        }
        os.read_ptr(file_handle, layers[0].biases.data, int(ggml.nbytes(layers[0].biases)))
    }

    // FC2
    {
        dimension_count: i32
        os.read_ptr(file_handle, &dimension_count, size_of(i32))

        ne_weight := [2]i32{1, 1}
        for i in 0 ..< dimension_count {
            os.read_ptr(file_handle, &ne_weight[i], size_of(i32))
        }
        os.read_ptr(file_handle, layers[1].weights.data, int(ggml.nbytes(layers[1].weights)))

        ne_bias := [2]i32{1, 1}
        for i in 0 ..< dimension_count {
            os.read_ptr(file_handle, &ne_bias[i], size_of(i32))
        }
        os.read_ptr(file_handle, layers[1].biases.data, int(ggml.nbytes(layers[1].biases)))
    }

    return true
}

// load_digit_png :: proc() -> (res: [INPUT_SIZE]f32) {
//     digit_image, err := png.load_from_file("digit.png")
//     if err != nil {
//         fmt.eprintln("Failed to load digit.png")
//     }
//     pixels := mem.slice_data_cast([]image.RGBA_Pixel, digit_image.pixels.buf[:])
//     for i in 0 ..< INPUT_SIZE {
//         p := pixels[i]
//         average_rgb := (f32(p.r) + f32(p.g) + f32(p.b)) / (255.0 * 3.0)
//         res[i] = average_rgb * f32(p.a) / 255.0
//     }
//     return
// }