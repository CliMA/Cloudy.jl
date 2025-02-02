import OrdinaryDiffEq as ODE
import ClimaCore as CC
import ClimaParams as CP
import CloudMicrophysics.Parameters as CMP
import KinematicDriver
import KinematicDriver.Common as CO
import KinematicDriver.K1DModel as K1D

include(joinpath(pkgdir(KinematicDriver), "test", "create_parameters.jl"))
include(joinpath(pkgdir(KinematicDriver), "test", "plotting_utils.jl"))

function run_KiD_col_sed_simulation(::Type{FT}, opts) where {FT}

    # Equations to solve for precipitation variables
    precipitation_choice = opts["precipitation_choice"]
    rain_formation_choice = opts["rain_formation_choice"]
    sedimentation_choice = opts["sedimentation_choice"]
    @info precipitation_choice, rain_formation_choice, sedimentation_choice

    # Decide the output flder name based on options
    output_folder = string("Output_", precipitation_choice)
    if precipitation_choice in ["Precipitation1M", "Precipitation2M"]
        output_folder = output_folder * "_" * rain_formation_choice
        if sedimentation_choice == "Chen2022"
            output_folder = output_folder * "_Chen2022"
        end
    elseif precipitation_choice == "CloudyPrecip"
        output_folder = output_folder * "_" * string(opts["num_moments"])
    end
    path = joinpath(@__DIR__, output_folder)
    mkpath(path)

    # Overwrite the defaults parameters based on options
    default_toml_dict = CP.create_toml_dict(FT)
    toml_dict = override_toml_dict(
        path,
        default_toml_dict,
        precip_sources = 1,
        precip_sinks = 0,
        prescribed_Nd = FT(opts["prescribed_Nd"]),
    )
    toml_dict["SB2006_cloud_gamma_distribution_parameter"]["value"] = opts["k"]
    # Create Thermodynamics.jl and KinematicDriver model parameters
    # (some of the CloudMicrophysics.jl parameters structs are created later based on model choices)
    common_params = create_common_parameters(toml_dict)
    kid_params = create_kid_parameters(toml_dict)
    thermo_params = create_thermodynamics_parameters(toml_dict)
    air_params = CMP.AirProperties(toml_dict)
    activation_params = CMP.AerosolActivationParameters(toml_dict)

    moisture_choice = precipitation_choice == "CloudyPrecip" ? "CloudyMoisture" : "NonEquilibriumMoisture"
    moisture = CO.get_moisture_type(moisture_choice, toml_dict)
    precip = CO.get_precipitation_type(
        precipitation_choice,
        toml_dict,
        rain_formation_choice = rain_formation_choice,
        sedimentation_choice = sedimentation_choice,
    )

    @info "Initialising"
    # Initialize the timestepping struct
    TS = CO.TimeStepping(FT(opts["dt"]), FT(opts["dt_output"]), FT(opts["t_end"]))

    # Create the coordinates
    space, face_space = K1D.make_function_space(FT, FT(opts["z_min"]), FT(opts["z_max"]), opts["n_elem"])
    coord = CC.Fields.coordinate_field(space)
    face_coord = CC.Fields.coordinate_field(face_space)

    # Initialize the netcdf output Stats struct
    fname = joinpath(path, "Output.nc")
    Stats = CO.NetCDFIO_Stats(
        fname,
        1.0,
        parent(face_coord),
        parent(coord),
        output_profiles = Dict(
            :ρ => "density",
            :q_tot => "q_tot",
            :q_liq => "q_liq",
            :q_rai => "q_rai",
            :N_liq => "N_liq",
            :N_rai => "N_rai",
        ),
    )

    # Create the initial condition profiles
    if precipitation_choice == "CloudyPrecip"
        cloudy_disttypes = determine_cloudy_disttypes(opts["num_moments"])
        cloudy_params, cloudy_pdists = create_cloudy_parameters(FT, cloudy_disttypes)
        init = map(
            coord -> CO.cloudy_initial_condition(
                cloudy_pdists,
                CO.initial_condition_0d(FT, thermo_params, opts["qt"], opts["prescribed_Nd"], opts["k"], opts["rhod"]),
                opts["k"],
            ),
            coord,
        )
    else
        cloudy_params = nothing
        init = map(
            coord -> CO.initial_condition_0d(
                FT,
                thermo_params,
                opts["qt"],
                opts["prescribed_Nd"],
                opts["k"],
                opts["rhod"],
            ),
            coord,
        )
    end

    # Create aux vector and apply initial condition
    aux = K1D.initialise_aux(
        FT,
        init,
        common_params,
        kid_params,
        thermo_params,
        air_params,
        activation_params,
        TS,
        Stats,
        face_space,
        moisture,
        precip,
        cloudy_params,
    )

    # Create state vector and apply initial condition
    Y = CO.initialise_state(moisture, precip, init)

    # Output the initial condition
    CO.simulation_output(aux, 0.0)

    # Define callbacks for output
    callback_io = ODE.DiscreteCallback(CO.condition_io, CO.affect_io!; save_positions = (false, false))
    callbacks = ODE.CallbackSet(callback_io)

    # Collect all the tendencies into rhs function for ODE solver
    # based on model choices for the solved equations
    ode_rhs! = K1D.make_rhs_function_col_sed(moisture, precip)

    # Solve the ODE operator
    problem = ODE.ODEProblem(ode_rhs!, Y, (FT(opts["t_ini"]), FT(opts["t_end"])), aux)
    @info "Solving"
    solver = ODE.solve(
        problem,
        ODE.SSPRK33(),
        dt = TS.dt,
        saveat = TS.dt_io,
        callback = callbacks,
        progress = true,
        progress_message = (dt, u, p, t) -> t,
    )

    # Some basic plots
    @info "Plotting"
    plot_folder = joinpath(path, "figures")
    plot_timeheight_no_ice_snow(joinpath(path, "Output.nc"), output = plot_folder)
end

opts = Dict(
    "qt" => 1e-3,
    "prescribed_Nd" => 1e8,
    "k" => 2.0,
    "rhod" => 1.0,
    "precipitation_choice" => "CloudyPrecip",
    "num_moments" => 6,
    "rain_formation_choice" => "CliMA_1M",
    "sedimentation_choice" => "CliMA_1M",
    "z_min" => 0.0,
    "z_max" => 3000.0,
    "n_elem" => 60,
    "dt" => 1.0,
    "dt_output" => 30.0,
    "t_ini" => 0.0,
    "t_end" => 3600.0,
)
run_KiD_col_sed_simulation(Float64, opts);
