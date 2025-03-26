#include "BinaryBHLevel.hpp"

#include "PositiveChiAndAlpha.hpp"
#include "TraceARemoval.hpp"

// Calculate RHS during RK4 substeps
void BinaryBHLevel::specificEvalRHS(amrex::MultiFab &a_soln,
                                    amrex::MultiFab &a_rhs,
                                    const double /*a_time*/)
{
    BL_PROFILE("BinaryBHLevel::specificEvalRHS()");
    const auto &soln_arrs   = a_soln.arrays();
    const auto &soln_c_arrs = a_soln.const_arrays();
    const auto &rhs_arrs    = a_rhs.arrays();

    // Enforce positive chi and alpha and trace free A
    amrex::ParallelFor(a_soln, a_soln.nGrowVect(),
                       [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k)
                       {
                           amrex::CellData<amrex::Real> cell =
                               soln_arrs[box_no].cellData(i, j, k);
                           TraceARemoval()(cell);
                           PositiveChiAndAlpha()(cell);
                       });

    // Calculate CCZ4 right hand side
    if (simParams().max_spatial_derivative_order == 4)
    {
	if (m_ccz4rhs.empty()) {
	    CCZ4RHS<MovingPunctureGauge, FourthOrderDerivatives> ccz4rhs(
		simParams().ccz4_params, Geom().CellSize(0), simParams().sigma,
		simParams().formulation);
	    m_ccz4rhs.resize(1);
	    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
				  &ccz4rhs, &ccz4rhs+1, m_ccz4rhs.begin());
	    amrex::Gpu::streamSynchronize();
	}

	amrex::Gpu::streamSynchronize();
	BL_PROFILE("KERNEL");

	auto* ccz4rhs = m_ccz4rhs.data();
        amrex::ParallelFor(a_rhs,
                           [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k)
                           {
                               ccz4rhs->compute(i, j, k, rhs_arrs[box_no],
						soln_c_arrs[box_no]);
                           });

	amrex::Gpu::streamSynchronize();
    }
    else if (simParams().max_spatial_derivative_order == 6)
    {
        amrex::Abort("xxxxx max_spatial_derivative_order == 6 todo");
#if 0
        CCZ4RHS<MovingPunctureGauge, SixthOrderDerivatives>
            ccz4rhs(simParams().ccz4_params, Geom().CellSize(0), simParams().sigma,
                    simParams().formulation);
        amrex::ParallelFor(a_rhs,
        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k)
        {
            amrex::CellData<amrex::Real const> state = soln_c_arrs[box_no].cellData(i,j,k);
            amrex::CellData<amrex::Real> rhs = rhs_arrs[box_no].cellData(i,j,k);
            ccz4rhs.compute(rhs, state);
        });
#endif
    }

    amrex::Gpu::streamSynchronize();
}
