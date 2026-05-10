const HAS_CUDA_TEST_DEP = !isnothing(Base.find_package("CUDA"))

if HAS_CUDA_TEST_DEP
    import CUDA
end

cuda_is_available() = HAS_CUDA_TEST_DEP && CUDA.functional()

function relative_max_error(a, b)
    return maximum(abs.(a .- b)) / max(maximum(abs, b), eps(real(eltype(b))))
end

@testset "CUDA spectral analysis" begin
    if cuda_is_available()
        CUDA.allowscalar(false)

        @testset "2D spectra match CPU" begin
            n = 32
            X, K, _, _ = xk_arrays((1.0, 1.0), (n, n))
            psi_cpu = Psi(nonlinear_field_2d(X), X, K)
            k = collect(LinRange(0.0, maximum(abs, K[1]), 24))
            psi_gpu = gpu(psi_cpu)
            cache = spectrum_cache(psi_gpu; k)

            analyze_spectra!(
                cache,
                psi_gpu;
                spectra = (:density, :kinetic, :incompressible, :compressible),
            )
            CUDA.synchronize()
            result = spectrum_results(cache; host = true)

            @test relative_max_error(result.density, density_spectrum(k, psi_cpu)) < 1.0e-8
            @test relative_max_error(result.kinetic, kinetic_density(k, psi_cpu)) < 1.0e-8
            @test relative_max_error(
                result.incompressible,
                incompressible_spectrum(k, psi_cpu),
            ) < 1.0e-8
            @test relative_max_error(
                result.compressible,
                compressible_spectrum(k, psi_cpu),
            ) < 1.0e-8

            et, ei, ec = energydecomp(psi_cpu)
            @test result.totals.kinetic ≈ sum(et) rtol = 1.0e-8
            @test result.totals.incompressible ≈ sum(ei) rtol = 1.0e-8
            @test result.totals.compressible ≈ sum(ec) rtol = 1.0e-8
        end

        @testset "3D spectra match CPU" begin
            n = 12
            X, K, _, _ = xk_arrays((1.0, 1.0, 1.0), (n, n, n))
            psi_cpu = Psi(nonlinear_field_3d(X), X, K)
            k = collect(LinRange(0.0, maximum(abs, K[1]), 16))
            psi_gpu = gpu(psi_cpu)
            cache = spectrum_cache(psi_gpu; k)

            analyze_spectra!(
                cache,
                psi_gpu;
                spectra = (:density, :kinetic, :incompressible, :compressible),
            )
            CUDA.synchronize()
            result = spectrum_results(cache; host = true)

            @test relative_max_error(result.density, density_spectrum(k, psi_cpu)) < 1.0e-8
            @test relative_max_error(result.kinetic, kinetic_density(k, psi_cpu)) < 1.0e-8
            @test relative_max_error(
                result.incompressible,
                incompressible_spectrum(k, psi_cpu),
            ) < 1.0e-8
            @test relative_max_error(
                result.compressible,
                compressible_spectrum(k, psi_cpu),
            ) < 1.0e-8

            et, ei, ec = energydecomp(psi_cpu)
            @test result.totals.kinetic ≈ sum(et) rtol = 1.0e-8
            @test result.totals.incompressible ≈ sum(ei) rtol = 1.0e-8
            @test result.totals.compressible ≈ sum(ec) rtol = 1.0e-8
        end
    else
        @info "Skipping CUDA spectral-analysis tests because CUDA is unavailable"
    end
end
