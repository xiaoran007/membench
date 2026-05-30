#ifndef MEMBENCH_MEMORY_ALIGNED_BUFFER_H
#define MEMBENCH_MEMORY_ALIGNED_BUFFER_H

#include <cstddef>
#include <cstdint>

namespace membench {

class AlignedBuffer {
public:
    AlignedBuffer() = default;
    AlignedBuffer(std::size_t size, std::size_t alignment);

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& other) noexcept;
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept;

    ~AlignedBuffer();

    std::uint8_t* data() { return data_; }
    const std::uint8_t* data() const { return data_; }
    std::size_t size() const { return size_; }

private:
    void reset();

    std::uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t alignment_ = 0;
};

}  // namespace membench

#endif  // MEMBENCH_MEMORY_ALIGNED_BUFFER_H
