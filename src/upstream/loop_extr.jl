"""
Helper function gives genomic length of loop (a,b) on
a circular chromosome of length N
"""
d_periodic(a,b,N)=min(abs(a-b),N-abs(a-b))

"""
Given pair-wise contact frequencies P(i,j)
and the non-interacting background P_0(i,j)
Return 3D prediction for contact triplets

    P(i,j,k) ≈ P(i,j)P(j,k)+
    P(j,k)P(k,i)+
    P(k,i)P(i,j)-
    2P(i,j)P(j,k)P(k,i)

"""
function loop_extr(contacts;periodic=true)
    N=size(contacts)[1]
    triplets=zeros(N,N,N)
    for i in 1:N
        for j in 1:N
            for k in 1:N
                if periodic
                    if d_periodic(i,j,N)>1 && d_periodic(j,k,N)>1 && d_periodic(i,k,N)>1
                        p=min(1,contacts[i,j]*contacts[j,k]+contacts[i,k]*contacts[k,j]+contacts[j,i]*contacts[i,k]-2*contacts[i,j]*contacts[j,k]*contacts[k,i])
                        triplets[k,j,i]=p
                    end
                else
                    if i!=j && k!=i && j!=k
                        p=min(1,contacts[i,j]*contacts[j,k]+contacts[i,k]*contacts[k,j]+contacts[j,i]*contacts[i,k]-2*contacts[i,j]*contacts[j,k]*contacts[k,i])
                        triplets[k,j,i]=p
                    end
                end
            end
        end
    end
    return triplets
end
