//
// Created by galls2 on 07/09/19.
//
#pragma once
#include <unordered_map>
#include <map>
#include <vector>
#include <string>
#include <memory>
#include <experimental/optional>
#include <z3++.h>
#include <bits/unordered_map.h>

enum AigMetadata{
    M = 0, I = 1, L = 2, O = 3, A = 4
};


class AigParser {
public:

    explicit AigParser(const std::string& aig_path);
    const std::unordered_map<AigMetadata, size_t, std::hash<size_t>>& get_aig_metadata();

private:
    void extract_metadata(const std::string& first_aag_line);
    void extract_literals(const std::vector<std::string>& aag_lines);
    const AigParser& dfs(const std::vector<std::string> &lines, std::unordered_map<size_t, z3::expr>& formulas, size_t first_line, size_t target_lit) const;
    void extract_ap_mapping(const std::vector<std::string>& vector);
    std::unordered_map<size_t, z3::expr> calc_literal_formulas(const std::vector<std::string>& aag_lines);
    void calculate_tr_formula(const std::unordered_map<size_t, z3::expr>& fresh_formulas);
    void extract_init(const std::vector<std::string> &file_lines);
    void generate_new_names(std::vector<std::reference_wrapper<std::vector<z3::expr>>>& vec_of_vecs, size_t& first_name, size_t vars_per_vec);

    size_t _first_ap_index = 0;
    size_t _first_and_literal = 0;
    std::unordered_map<AigMetadata, size_t, std::hash<size_t>> _metadata;
    std::unordered_map<std::string, std::string> _ap_to_symb;
    std::unordered_map<std::string, std::string> _symb_to_ap;

    std::vector<size_t> _in_literals;
    std::vector<size_t> _out_literals;
    std::vector<size_t> _prev_state_literals;
    std::vector<size_t> _next_state_literals;
//    std::unordered_map<size_t, z3::expr> _lit_formulas;
    z3::context _ctx;
    std::unique_ptr<PropFormula> _tr_formula;
    std::unique_ptr<z3::expr> _init_formula;
    std::unique_ptr<z3::expr>  _state_formula;

    void generate_state_formula(const std::unordered_map<size_t, z3::expr> &formulas, std::vector<z3::expr> &prev_out,
                                const z3::expr_vector &orig_in, const z3::expr_vector &orig_ps,
                                std::vector<z3::expr> &prev_in, std::vector<z3::expr> &prev_latch);
};
