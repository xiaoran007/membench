#include "memory/aligned_buffer.h"

#include <cstdlib>
#include <stdexcept>

#ifdef _WIN32
#include <malloc.h>
#endif

namespace membench {

AlignedBuffer::AlignedBuffer(std::size_t size, std::size_t alignment)
    : size_(size), alignment_(alignment) {
    if (size_ == 0) {
        throw std::runtime_error("buffer size must be greater than zero");
    }

#ifdef _WIN32
    data_ = static_cast<std::uint8_t*>(_aligned_malloc(size_, alignment_));
    if (data_ == nullptr) {
        throw std::bad_alloc();
    }
#else
    void* raw = nullptr;
    if (posix_memalign(&raw, alignment_, size_) != 0 || raw == nullptr) {
        throw std::bad_alloc();
    }
    data_ = static_cast<std::uint8_t*>(raw);
#endif
}

AlignedBuffer::AlignedBuffer(AlignedBuffer&& other) noexcept
    : data_(other.data_), size_(other.size_), alignment_(other.alignment_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.alignment_ = 0;
}

AlignedBuffer& AlignedBuffer::operator=(AlignedBuffer&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    reset();
    data_ = other.data_;
    size_ = other.size_;
    alignment_ = other.alignment_;
    other.data_ = nullptr;
    other.size_ = 0;
    other.alignment_ = 0;
    return *this;
}

AlignedBuffer::~AlignedBuffer() {
    reset();
}

void AlignedBuffer::reset() {
    if (data_ == nullptr) {
        return;
    }
#ifdef _WIN32
    _aligned_free(data_);
#else
    free(data_);
#endif
    data_ = nullptr;
    size_ = 0;
    alignment_ = 0;
}

}  // namespace membench
