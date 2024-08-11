set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_definitions(/DNDEBUG)
endif()

include(CheckSymbolExists)

if (CMAKE_CXX_COMPILER_ID MATCHES GNU)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-strict-aliasing -fPIC")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fexceptions -fno-rtti -Wno-strict-aliasing -Wno-class-memaccess -fPIC")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -s")

    if (XMRIG_ARMv8)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ARM8_CXX_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARM8_CXX_FLAGS} -flax-vector-conversions")
    elseif (XMRIG_ARMv7)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfpu=neon")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon -flax-vector-conversions")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -maes")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -maes")

        add_definitions(/DHAVE_ROTR)
    endif()

    if (WIN32)
        if (CMAKE_SIZEOF_VOID_P EQUAL 8)
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
        else()
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static -Wl,--large-address-aware")
        endif()
    else()
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")
    endif()

    add_definitions(/D_GNU_SOURCE)

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

    add_definitions(/DHAVE_BUILTIN_CLEAR_CACHE)

elseif (CMAKE_CXX_COMPILER_ID MATCHES MSVC)

    set(CMAKE_C_FLAGS_RELEASE "/MP /MT /O2 /Ob2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "/MP /MT /O2 /Ob2 /DNDEBUG")
    add_definitions(/D_CRT_SECURE_NO_WARNINGS)
    add_definitions(/D_CRT_NONSTDC_NO_WARNINGS)
    add_definitions(/DNOMINMAX)
    add_definitions(/DHAVE_ROTR)

elseif (CMAKE_CXX_COMPILER_ID MATCHES Clang)

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -fPIC")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2 -funroll-loops -fmerge-all-constants")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fexceptions -fno-rtti -Wno-missing-braces -fPIC")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -funroll-loops -fmerge-all-constants")

    if (XMRIG_ARMv8)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ARM8_CXX_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARM8_CXX_FLAGS}")
    elseif (XMRIG_ARMv7)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfpu=neon -march=${CMAKE_SYSTEM_PROCESSOR}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon -march=${CMAKE_SYSTEM_PROCESSOR}")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -maes")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -maes")

        check_symbol_exists("_rotr" "x86intrin.h" HAVE_ROTR)
        if (HAVE_ROTR)
            add_definitions(/DHAVE_ROTR)
        endif()
    endif()

    if (XMRIG_OS_APPLE)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    endif()

endif()

if (NOT WIN32)
    check_symbol_exists("__builtin___clear_cache" "stdlib.h" HAVE_BUILTIN_CLEAR_CACHE)
    if (HAVE_BUILTIN_CLEAR_CACHE)
        add_definitions(/DHAVE_BUILTIN_CLEAR_CACHE)
    endif()
endif()
