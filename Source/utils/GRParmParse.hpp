/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef GRPARMPARSE_HPP_
#define GRPARMPARSE_HPP_

#include "ArrayTools.hpp"

#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <memory>
#include <type_traits>

class GRParmParse : public amrex::ParmParse
{
  public:
    using amrex::ParmParse::ParmParse; // Just use ParmParse's constructor

    // (MK): I called the functions below "load" rather than "get" to avoid
    // clashes with the many  different overloads of "get" in ParmParse. Also, I
    // think load is a more intuitive name.

    /// Loads an array from the parameter file
    template <class data_t, long unsigned int n_comp>
    void load(const char *name, std::array<data_t, n_comp> &array) const
    {
        get<data_t, n_comp>(name, array);
    }

    template <long unsigned int n_comp>
    void load(const char *name, std::array<bool, n_comp> &array) const
    {
        std::array<int, n_comp> tmp{};
        get<int, n_comp>(name, tmp);
        for (long unsigned int i = 0; i < n_comp; ++i)
        {
            array[i] = tmp[i];
        }
    }

    /// Loads a vector with num_comp components from the parameter file
    template <class data_t>
    void load(const char *name, std::vector<data_t> &vector,
              const int num_comp) const
    {
        getarr(name, vector, 0, num_comp);
    }

    template <class data_t>
    void load(const char *name, std::vector<bool> &vector,
              const int num_comp) const
    {
        std::vector<int> tmp(num_comp);
        getarr(name, tmp, 0, num_comp);
        for (long unsigned int i = 0; i < num_comp; ++i)
        {
            vector[i] = tmp[i];
        }
    }

    /// Loads a value from the parameter file
    template <class data_t>
    typename std::enable_if_t<
        !std::is_enum_v<data_t>> // Can't use for enum types
    load(const char *name, data_t &parameter) const
    {
        get(name, parameter);
    }

    /// Loads an enum value from the parameter file
    template <typename enum_type>
    typename std::enable_if_t<
        std::is_enum_v<enum_type>> // Only enabled for enum types
    load(const char *name, enum_type &parameter) const
    {
        int iparam = 0;
        get(name, iparam);
        parameter = static_cast<enum_type>(iparam);
    }

    /// Loads a value from the parameter file, if the value isn't defined it
    /// sets to the supplied default
    template <class data_t>
    void load(const char *name, data_t &parameter,
              const data_t default_value) const
    {
        if (contains(name))
        {
            load(name, parameter);
        }
        else
        {
            parameter = default_value;
            default_message(name, default_value);
        }
    }

    /// Loads a vector with num_comp components from the parameter file, if the
    /// vector isn't defined, it is set to the supplied default
    template <class data_t>
    void load(const char *name, std::vector<data_t> &vector, const int num_comp,
              const std::vector<data_t> &default_vector) const
    {
        if (contains(name))
        {
            load(name, vector, num_comp);
        }
        else
        {
            vector = default_vector;
            default_message(name, default_vector);
        }
    }

    /// Loads a vector with num_comp components from the parameter file, if the
    /// vector isn't defined it sets all components to the supplied default
    template <class data_t>
    void load(const char *name, std::vector<data_t> &vector, const int num_comp,
              const data_t default_value) const
    {
        load(name, vector, num_comp,
             std::vector<data_t>(num_comp, default_value));
    }

  protected:
    template <typename data_t,
              std::enable_if_t<
                  !ArrayTools::is_std_array_or_vector<data_t>::value,
                  bool> = true> // this won't work for std::arrays and vectors
    void default_message(const char *name, const data_t &default_value) const
    {
        amrex::Print() << "Parameter: " << name
                       << " not found in parameter file. "
                       << "It has been set to its default value = "
                       << default_value << "." << std::endl;
    }

    template <typename data_t,
              std::enable_if_t<
                  ArrayTools::is_std_array_or_vector<data_t>::value,
                  bool> = true> // use this code for std::arrays and vectors
    void default_message(const char *name, const data_t &default_value) const
    {
        amrex::Print() << "Parameter: " << name
                       << " not found in parameter file. "
                       << "It has been set to its default "
                          "value =";
        for (const auto &elem : default_value)
        {
            amrex::Print() << " " << elem;
        }
        amrex::Print() << "." << '\n';
    }
};

#endif /* GRPARMPARSE_HPP_ */
