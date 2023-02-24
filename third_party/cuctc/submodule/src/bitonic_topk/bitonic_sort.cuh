/**
 *  Modified from Rapidsai/raft(https://github.com/rapidsai/raft)
 * 
 */

#pragma once
#include <cstdint>
#include <type_traits>
namespace cu_ctc
{

    namespace topk
    {
        static constexpr int WarpSize = 32;
        template <typename IntType>
        constexpr inline __host__ __device__ bool isPo2(IntType num)
        {
            return (num && !(num & (num - 1)));
        }

        inline __device__ int laneId()
        {
            int id;
            asm("mov.s32 %0, %%laneid;"
                : "=r"(id));
            return id;
        }
        /**
         * @brief Shuffle the data inside a warp
         * @tparam T the data type (currently assumed to be 4B)
         * @param val value to be shuffled
         * @param laneMask mask to be applied in order to perform xor shuffle
         * @param width lane width
         * @param mask mask of participating threads (Volta+)
         * @return the shuffled data
         */
        template <typename T>
        inline __device__ T shfl_xor(T val, int laneMask, int width = WarpSize, uint32_t mask = 0xffffffffu)
        {
#if CUDART_VERSION >= 9000
            return __shfl_xor_sync(mask, val, laneMask, width);
#else
            return __shfl_xor(val, laneMask, width);
#endif
        }

        /**
         * @brief Shuffle the data inside a warp
         * @tparam T the data type (currently assumed to be 4B)
         * @param val value to be shuffled
         * @param srcLane lane from where to shuffle
         * @param width lane width
         * @param mask mask of participating threads (Volta+)
         * @return the shuffled data
         */
        template <typename T>
        inline __device__ T shfl(T val, int srcLane, int width = WarpSize, uint32_t mask = 0xffffffffu)
        {
#if CUDART_VERSION >= 9000
            return __shfl_sync(mask, val, srcLane, width);
#else
            return __shfl(val, srcLane, width);
#endif
        }

        /** warp-wide any boolean aggregator */
        inline __device__  bool any(bool inFlag, uint32_t mask = 0xffffffffu)
        {
#if CUDART_VERSION >= 9000
            inFlag = __any_sync(mask, inFlag);
#else
            inFlag = __any(inFlag);
#endif
            return inFlag;
        }

        template <typename T>
        constexpr T lower_bound()
        {
            if constexpr (std::numeric_limits<T>::has_infinity && std::numeric_limits<T>::is_signed)
            {
                return -std::numeric_limits<T>::infinity();
            }
            return std::numeric_limits<T>::lowest();
        }

        template <typename T>
        constexpr T upper_bound()
        {
            if constexpr (std::numeric_limits<T>::has_infinity)
            {
                return std::numeric_limits<T>::infinity();
            }
            return std::numeric_limits<T>::max();
        }

        namespace helpers
        {

            template <typename T>
            __device__ __forceinline__ void swap(T &x, T &y)
            {
                T t = x;
                x = y;
                y = t;
            }

            template <typename T>
            __device__ __forceinline__ void conditional_assign(bool cond, T &ptr, T x)
            {
                if (cond)
                {
                    ptr = x;
                }
            }

        } // namespace helpers

        /**
         * Warp-wide bitonic merge and sort.
         * The data is strided among `warp_width` threads,
         * e.g. calling `bitonic<4>(ascending=true).sort(arr)` takes a unique 4-element array as input of
         * each thread in a warp and sorts them, such that for a fixed i, arr[i] are sorted within the
         * threads in a warp, and for any i < j, arr[j] in any thread is not smaller than arr[i] in any
         * other thread.
         * When `warp_width < WarpSize`, the data is sorted within all subwarps of the warp independently.
         *
         * As an example, assuming `Size = 4`, `warp_width = 16`, and `WarpSize = 32`, sorting a permutation
         * of numbers 0-63 in each subwarp yield the following result:
         * `
         *  arr_i \ laneId()
         *       0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15    16  17  18 ...
         *      subwarp_1                                                         subwarp_2
         *   0   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15     0   1   2 ...
         *   1  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31    16  17  18 ...
         *   2  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47    32  33  34 ...
         *   3  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63    48  49  50 ...
         * `
         *
         * @tparam Size
         *   number of elements processed in each thread;
         *   i.e. the total data size is `Size * warp_width`.
         *   Must be power-of-two.
         *
         */
        template <int Size = 1>
        class bitonic
        {
            static_assert(isPo2(Size), "class bitonic<Size> , size should be power of 2 \n");

        public:
            /**
             * Initialize bitonic sort config.
             *
             * @param ascending
             *   the resulting order (true: ascending, false: descending).
             * @param warp_width
             *   the number of threads participating in the warp-level primitives;
             *   the total size of the sorted data is `Size * warp_width`.
             *   Must be power-of-two, not larger than the WarpSize.
             */
            __device__ __forceinline__ explicit bitonic(bool ascending, int warp_width = WarpSize)
                : ascending_(ascending), warp_width_(warp_width)
            {
            }

            bitonic(bitonic const &) = delete;
            bitonic(bitonic &&) = delete;
            auto operator=(bitonic const &) -> bitonic & = delete;
            auto operator=(bitonic &&) -> bitonic & = delete;

            /**
             * You can think of this function in two ways:
             *
             *   1) Sort any bitonic sequence.
             *   2) Merge two halfs of the input data assuming they're already sorted, and their order is
             *      opposite (i.e. either ascending, descending or vice-versa).
             *
             * The input pointers are unique per-thread.
             * See the class description for the description of the data layout.
             *
             * @param keys
             *   is a device pointer to a contiguous array of keys, unique per thread; must be at least `Size`
             *   elements long.
             * @param payloads
             *   are zero or more associated arrays of the same size as keys, which are sorted together with
             *   the keys; must be at least `Size` elements long.
             */
            template <typename KeyT, typename... PayloadTs>
            __device__ __forceinline__ void merge(KeyT *__restrict__ keys,
                                                  PayloadTs *__restrict__... payloads) const
            {
                return bitonic<Size>::merge_(ascending_, warp_width_, keys, payloads...);
            }

            /**
             * Sort the data.
             * The input pointers are unique per-thread.
             * See the class description for the description of the data layout.
             *
             * @param keys
             *   is a device pointer to a contiguous array of keys, unique per thread; must be at least `Size`
             *   elements long.
             * @param payloads
             *   are zero or more associated arrays of the same size as keys, which are sorted together with
             *   the keys; must be at least `Size` elements long.
             */
            template <typename KeyT, typename... PayloadTs>
            __device__ __forceinline__ void sort(KeyT *__restrict__ keys,
                                                 PayloadTs *__restrict__... payloads) const
            {
                return bitonic<Size>::sort_(ascending_, warp_width_, keys, payloads...);
            }

            /**
             * @brief `merge` variant for the case of one element per thread
             *        (pass input by a reference instead of a pointer).
             *
             * @param key
             * @param payload
             */
            template <typename KeyT, typename... PayloadTs, int S = Size>
            __device__ __forceinline__ auto merge(KeyT &__restrict__ key,
                                                  PayloadTs &__restrict__... payload) const
                -> std::enable_if_t<S == 1, void> // SFINAE to enable this for Size == 1 only
            {
                static_assert(S == Size);
                return merge(&key, &payload...);
            }

            /**
             * @brief `sort` variant for the case of one element per thread
             *        (pass input by a reference instead of a pointer).
             *
             * @param key
             * @param payload
             */
            template <typename KeyT, typename... PayloadTs, int S = Size>
            __device__ __forceinline__ auto sort(KeyT &__restrict__ key,
                                                 PayloadTs &__restrict__... payload) const
                -> std::enable_if_t<S == 1, void> // SFINAE to enable this for Size == 1 only
            {
                static_assert(S == Size);
                return sort(&key, &payload...);
            }

        private:
            const int warp_width_;
            const bool ascending_;

            template <int AnotherSize>
            friend class bitonic;

            template <typename KeyT, typename... PayloadTs>
            static __device__ __forceinline__ void merge_(bool ascending,
                                                          int warp_width,
                                                          KeyT *__restrict__ keys,
                                                          PayloadTs *__restrict__... payloads)
            {
#pragma unroll
                for (int size = Size; size > 1; size >>= 1)
                {
                    const int stride = size >> 1;
#pragma unroll
                    for (int offset = 0; offset < Size; offset += size)
                    {
#pragma unroll
                        for (int i = offset + stride - 1; i >= offset; i--)
                        {
                            const int other_i = i + stride;
                            KeyT &key = keys[i];
                            KeyT &other = keys[other_i];
                            if (ascending ? key > other : key < other)
                            {
                                helpers::swap(key, other);
                                (helpers::swap(payloads[i], payloads[other_i]), ...);
                            }
                        }
                    }
                }
                const int lane = laneId();
#pragma unroll
                for (int i = 0; i < Size; i++)
                {
                    KeyT &key = keys[i];
                    for (int stride = (warp_width >> 1); stride > 0; stride >>= 1)
                    {
                        const bool is_second = lane & stride;
                        const KeyT other = shfl_xor(key, stride, warp_width);
                        const bool do_assign = (ascending != is_second) ? key > other : key < other;

                        helpers::conditional_assign(do_assign, key, other);
                        // NB: don't put shfl_xor in a conditional; it must be called by all threads in a warp.
                        (helpers::conditional_assign(
                             do_assign, payloads[i], shfl_xor(payloads[i], stride, warp_width)),
                         ...);
                    }
                }
            }

            template <typename KeyT, typename... PayloadTs>
            static __device__ __forceinline__ void sort_(bool ascending,
                                                         int warp_width,
                                                         KeyT *__restrict__ keys,
                                                         PayloadTs *__restrict__... payloads)
            {
                if constexpr (Size == 1)
                {
                    const int lane = laneId();
                    for (int width = 2; width < warp_width; width <<= 1)
                    {
                        bitonic<1>::merge_(lane & width, width, keys, payloads...);
                    }
                }
                else
                {
                    constexpr int kSize2 = Size / 2;
                    bitonic<kSize2>::sort_(false, warp_width, keys, payloads...);
                    bitonic<kSize2>::sort_(true, warp_width, keys + kSize2, (payloads + kSize2)...);
                }
                bitonic<Size>::merge_(ascending, warp_width, keys, payloads...);
            }
        };
    } // namespace topk
} // namespace cu_ctc
