#include <RcppArmadillo.h>
#include <cmath>
#include <limits>
#include <time.h>
#include <algorithm>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// [[Rcpp::plugins("cpp11")]]

// [[Rcpp::export]]
arma::vec InbreedingSI(const arma::umat & Tmp){
  arma::uword i   = 0;
  arma::sword j   = 0;
  arma::sword t   = -1;
  arma::uword n   = Tmp.n_rows;
  arma::umat  Ped(n + 1, 3);
  arma::uword s   = 0;
  arma::uword d   = 0;
  arma::vec   F   = zeros(n + 1, 1);
  arma::vec   L   = zeros(n + 1, 1);
  arma::vec   B   = zeros(n + 1, 1);
  arma::uvec  ANC = zeros<uvec>(n + 1, 1);
  arma::ivec  LAP = zeros<ivec>(n + 1, 1);
  Ped.zeros();
  Ped(span(1, n), span(0, 2)) = Tmp;
  F(0)            = -1.0;
  LAP(0)          = -1;
  for(i = 1; i <= n; ++i){
    s       = Ped(i, 1);
    d       = Ped(i, 2);
    LAP(i)  = (LAP(s) < LAP(d) ? LAP(d) : LAP(s)) + 1;
    if (LAP[i] > t) t = LAP[i];
  }
  arma::uvec SI = zeros<uvec>(t + 1, 1);
  arma::uvec MI = zeros<uvec>(t + 1, 1);
  for(i = 1; i <=n; i++){
    s       = Ped(i, 1);
    d       = Ped(i, 2);
    B(i)    = 0.5 - 0.25*(F(s) + F(d));
    for(j = 0; j < LAP(i); ++j){
      ++SI(j); 
      ++MI(j);
    }
    if(s == 0 || d == 0){
      F(i) = 0;
      L(i) = 0;
      continue;
    }
    if((Ped(i - 1, 1) == s) && (Ped(i - 1, 2) == d)){
      F(i) = F(i - 1);
      L(i) = 0.0;
      continue;
    }
    F(i)          = -1.0;
    L(i)          = -1.0;
    t             = LAP(i);
    ANC(MI(t)++)  = i;
    while(t > -1){
      j = ANC(--MI(t));
      s = Ped(j, 1);
      d = Ped(j, 2);
      if(s != 0){
        if(L(s) == 0.0){
          ANC(MI(LAP(s))++) = s;
        }
        L(s) = L(s) + 0.5*L(j);
      }
      if(d != 0){
        if(L(d) == 0.0){
          ANC(MI(LAP(d))++) = d;
        }
        L(d) = L(d) + 0.5*L(j);
      }
      F(i) = F(i) + pow(L(j), 2.0)*B(j);
      L(j) = 0.0;
      if(MI(t) == SI(t)){
        --t;
      } 
    }
  }
  return(F.tail(n));
}

// [[Rcpp::export]]
void A_times_v(arma::vec &w, const arma::vec & v, const arma::umat & Ped, const arma::vec & F,
               const int n){
  //arma::uword n   = w.n_rows;
  arma::uword s   = 0;
  arma::uword d   = 0;
  arma::sword i   = 0;
  arma::vec   q   = zeros(n, 1);
  double      tmp = 0.0;
  double      di  = 0.0;
  for(i = n - 1; i > -1; i--){
    q(i)  = q(i) + v(i);
    s     = Ped(i, 1);
    d     = Ped(i, 2);
    if(s != 0){
      q(s) = q(s) + 0.5*q(i);
    }
    if(d != 0){
      q(d) = q(d) + 0.5*q(i);
    }
  }
  for(i = 0; i < n; i++){
    s     = Ped(i, 1);
    d     = Ped(i, 2);
    di    = static_cast<double>(accu(find(Ped.row(i) == 0)) + 2)/4.0 - 0.25*(F(s) + F(d));
    tmp   = 0.0;
    if(s != 0){
      tmp = tmp + w(s);
    }
    if(d != 0){
      tmp = tmp + w(d);
    }
    w(i)  = 0.5*tmp;
    w(i)  = w(i) + di*q(i);
  }
}

// [[Rcpp::export]]
arma::mat getA22(const arma::umat &Ped, const arma::uword nGen, const int n, const arma::uvec & GenID,
                 const arma::vec & F){
  arma::mat A22 = zeros(nGen, nGen);
  arma::vec w   = zeros(n, 1);
  arma::vec v   = zeros(n, 1);
  for(uword i = 0; i < nGen; i++){
    v(GenID(i) - 1) = 1.0;
    A_times_v(w, v, Ped, F, n);
    A22.col(i) = w(GenID - 1);
  }
  return(A22);
}
